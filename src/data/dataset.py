# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.labels.term_bins import term_to_label


def load_label_map(label_map_path: Path) -> Dict[str, Any]:
    with label_map_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class JsonlOffsetDataset(Dataset):
    """用文件偏移量做索引，适合超大 jsonl（不把全量读进内存）"""

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        if not self.jsonl_path.exists():
            raise FileNotFoundError(self.jsonl_path)

        self.offsets: List[int] = []
        offset = 0
        with self.jsonl_path.open("rb") as f:
            for line in f:
                self.offsets.append(offset)
                offset += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self.offsets[idx]
        with self.jsonl_path.open("rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8").strip()
        return json.loads(line)


class CAILMultiTaskDataset(Dataset):
    """
    三任务数据：
    - accusation: multi-hot (float32)  [num_acc]
    - articles:   multi-hot (float32)  [num_art]
    - term:       int64               []
    """

    def __init__(
        self,
        jsonl_path: Path,
        label_map_path: Path,
        max_fact_chars: Optional[int] = None,
    ):
        self.base = JsonlOffsetDataset(jsonl_path)

        lm = load_label_map(label_map_path)
        self.acc2id = lm["accusation2id"]
        self.art2id = lm["article2id"]

        self.num_acc = len(self.acc2id)
        self.num_art = len(self.art2id)
        self.max_fact_chars = max_fact_chars

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.base[idx]
        fact = obj.get("fact", "")
        if self.max_fact_chars is not None:
            fact = fact[: self.max_fact_chars]

        meta = obj.get("meta", {})

        # accusation multi-hot
        y_acc = torch.zeros(self.num_acc, dtype=torch.float32)
        for a in meta.get("accusation", []):
            a = str(a)
            if a in self.acc2id:
                y_acc[self.acc2id[a]] = 1.0

        # articles multi-hot
        y_art = torch.zeros(self.num_art, dtype=torch.float32)
        for r in meta.get("relevant_articles", []):
            r = str(r)
            if r in self.art2id:
                y_art[self.art2id[r]] = 1.0

        # term single label
        term_obj = meta.get("term_of_imprisonment", {
            "death_penalty": False,
            "life_imprisonment": False,
            "imprisonment": 0
        })
        y_term = torch.tensor(term_to_label(term_obj), dtype=torch.long)

        return {
            "text": fact,
            "labels_accusation": y_acc,
            "labels_articles": y_art,
            "labels_term": y_term,
        }


from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class Collator:
    tokenizer: Any
    max_length: int = 512
    add_global_attention_mask: bool = False   # ✅ 新增：Longformer/Lawformer 才需要

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [x["text"] for x in batch]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=(256 if self.add_global_attention_mask else None),
            return_tensors="pt",
        )

        enc["labels_accusation"] = torch.stack([x["labels_accusation"] for x in batch], dim=0)
        enc["labels_articles"]   = torch.stack([x["labels_articles"] for x in batch], dim=0)
        enc["labels_term"]       = torch.stack([x["labels_term"] for x in batch], dim=0)

        # ✅ Longformer/Lawformer：给 CLS(第0个token) 开全局注意力
        if self.add_global_attention_mask:
            gam = torch.zeros_like(enc["attention_mask"])
            gam[:, 0] = 1
            enc["global_attention_mask"] = gam

        return enc
