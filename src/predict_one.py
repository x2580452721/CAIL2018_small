# -*- coding: utf-8 -*-
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

from src.models.multitask_head import MultiTaskClassifier
TERM_BINS = ["death", "life", "0", "1-6", "7-12", "13-36", "37-60", "61-120", ">120"]
def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "first_stage" if (project_root / "first_stage").exists() else (project_root / "data" / "first_stage")

    best_dir = project_root / "outputs" / "exp_roberta_baseline" / "best"
    encoder_dir = best_dir / "encoder"
    tok_dir = best_dir / "tokenizer"
    head_path = best_dir / "multitask_head.pt"

    label_map_path = data_root / "label_map.json"
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id2acc = {v: k for k, v in label_map["accusation2id"].items()}
    id2art = {v: k for k, v in label_map["article2id"].items()}

    tok = AutoTokenizer.from_pretrained(tok_dir)
    encoder = AutoModel.from_pretrained(encoder_dir)

    num_acc = len(label_map["accusation2id"])
    num_art = len(label_map["article2id"])
    num_term = len(label_map.get("term_bins", [])) or 9

    model = MultiTaskClassifier(encoder, encoder.config.hidden_size, num_acc, num_art, num_term, dropout=0.1)
    model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    text = input("输入案件事实 fact：\n").strip()
    enc = tok(text, max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        acc_logits, art_logits, term_logits = model(**enc)

    th = 0.20
    acc_prob = torch.sigmoid(acc_logits)[0].detach().cpu()
    art_prob = torch.sigmoid(art_logits)[0].detach().cpu()
    term_bin = int(term_logits.argmax(dim=-1)[0].item())
    term_text = TERM_BINS[term_bin]

    acc_ids = (acc_prob >= th).nonzero(as_tuple=True)[0].tolist()
    art_ids = (art_prob >= th).nonzero(as_tuple=True)[0].tolist()

    acc_top = sorted([(id2acc[i], float(acc_prob[i])) for i in acc_ids], key=lambda x: x[1], reverse=True)[:10]
    art_top = sorted([(id2art[i], float(art_prob[i])) for i in art_ids], key=lambda x: x[1], reverse=True)[:10]

    print("\n" + "=" * 52)
    print("司法要素预测结果（仅供辅助参考，不构成法律意见）")
    print("=" * 52)

    print("\n【输入事实摘要】")
    print(text[:200] + ("..." if len(text) > 200 else ""))

    print("\n【罪名 Top5】")
    for k, (name, p) in enumerate(acc_top, start=1):
        print(f"{k:>2}. {name:<10}  置信度={p:.4f}")

    print("\n【法条 Top5】")
    for k, (art, p) in enumerate(art_top, start=1):
        # 你的 label_map 里法条是字符串数字（例如 '234'）
        print(f"{k:>2}. 刑法第{art}条  置信度={p:.4f}")

    print("\n【刑期区间】")
    print(f"预测刑期：{term_text} （term_bin={term_bin}）")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()
