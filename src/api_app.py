# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

from src.models.multitask_head import MultiTaskClassifier

TERM_BINS = ["death", "life", "0", "1-6", "7-12", "13-36", "37-60", "61-120", ">120"]


def find_data_root(project_root: Path) -> Path:
    a = project_root / "data" / "first_stage"
    b = project_root / "first_stage"
    if (a / "label_map.json").exists():
        return a
    if (b / "label_map.json").exists():
        return b
    raise FileNotFoundError("Cannot find label_map.json in data/first_stage or first_stage")


def topk_always(prob_vec: torch.Tensor, id2label: dict, k: int = 5) -> List[Tuple[str, float]]:
    """永远返回 top-k（不做阈值筛选）"""
    k = min(k, prob_vec.numel())
    v, idx = torch.topk(prob_vec, k=k)
    return [(id2label[int(i)], float(p)) for p, i in zip(v.tolist(), idx.tolist())]


def threshold_then_topk(prob_vec: torch.Tensor, id2label: dict, th: float, k: int = 5) -> List[Tuple[str, float]]:
    """
    先阈值筛选（>=th），再按概率排序取前 k
    若一个都没过阈值，则兜底返回 top-k（避免空列表影响展示）
    """
    ids = (prob_vec >= th).nonzero(as_tuple=True)[0].tolist()
    pairs = [(id2label[i], float(prob_vec[i])) for i in ids]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if len(pairs) == 0:
        return topk_always(prob_vec, id2label, k)
    return pairs[:k]


# ---------- FastAPI ----------
app = FastAPI(title="CAIL2018 TextCls API", version="1.1")

# 允许网页直接访问（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictIn(BaseModel):
    fact: str
    max_len: int = 256

    # 新增：阈值 & 模式
    threshold: float = 0.20          # threshold 模式下生效
    mode: str = "threshold"          # "threshold" 或 "topk"
    topk: int = 5                    # topk 数量（两种模式都可用）


@app.get("/")
def home():
    return {"msg": "OK. Go to /docs to test the API."}


# ---------- 加载模型（只加载一次） ----------
project_root = Path(__file__).resolve().parents[1]
data_root = find_data_root(project_root)

best_dir = project_root / "outputs" / "exp_roberta_baseline" / "best"
encoder_dir = best_dir / "encoder"
tok_dir = best_dir / "tokenizer"
head_path = best_dir / "multitask_head.pt"

if not (encoder_dir.exists() and tok_dir.exists() and head_path.exists()):
    raise FileNotFoundError(f"Best model not found under: {best_dir}")

label_map_path = data_root / "label_map.json"
label_map = json.loads(label_map_path.read_text(encoding="utf-8"))

id2acc = {v: k for k, v in label_map["accusation2id"].items()}
id2art = {v: k for k, v in label_map["article2id"].items()}

num_acc = len(label_map["accusation2id"])
num_art = len(label_map["article2id"])
num_term = len(label_map.get("term_bins", [])) or 9

tokenizer = AutoTokenizer.from_pretrained(tok_dir)
encoder = AutoModel.from_pretrained(encoder_dir)

model = MultiTaskClassifier(
    encoder=encoder,
    hidden_size=encoder.config.hidden_size,
    num_acc=num_acc,
    num_art=num_art,
    num_term=num_term,
    dropout=0.1,
)
model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()


@app.post("/predict")
@torch.no_grad()
def predict(req: PredictIn) -> Dict[str, Any]:
    # --- 编码 ---
    enc = tokenizer(
        req.fact,
        max_length=req.max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # --- 推理 ---
    acc_logits, art_logits, term_logits = model(**enc)

    acc_prob = torch.sigmoid(acc_logits)[0].detach().cpu()
    art_prob = torch.sigmoid(art_logits)[0].detach().cpu()
    term_bin = int(term_logits.argmax(dim=-1)[0].item())

    # --- 输出策略 ---
    mode = (req.mode or "threshold").lower()
    th = float(req.threshold)
    k = int(req.topk)

    if mode == "topk":
        acc_list = topk_always(acc_prob, id2acc, k=k)
        art_list = topk_always(art_prob, id2art, k=k)
    else:
        # 默认：阈值筛选 + 为空兜底 topk
        acc_list = threshold_then_topk(acc_prob, id2acc, th=th, k=k)
        art_list = threshold_then_topk(art_prob, id2art, th=th, k=k)

    return {
        # 保持你原来的字段名，网页不用改
        "accusation_top5": acc_list,
        "articles_top5": [(f"刑法第{a}条", p) for a, p in art_list],
        "term_bin": term_bin,
        "term_range": TERM_BINS[term_bin],

        # 附加：告诉前端本次用了什么策略（不影响你原来的网页）
        "meta": {
            "mode": mode,
            "threshold": th,
            "topk": k,
            "max_len": req.max_len,
            "device": str(device),
        }
    }
