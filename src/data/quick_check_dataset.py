# -*- coding: utf-8 -*-
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import CAILMultiTaskDataset, Collator


def main():
    # 自动推断项目根目录：.../src/data/quick_check_dataset.py -> parents[2]
    root = Path(__file__).resolve().parents[2]

    train_path = root / "first_stage" / "train_split.json"
    label_map_path = root / "first_stage" / "label_map.json"

    print("train_path:", train_path)
    print("label_map_path:", label_map_path)

    # 先用 RoBERTa 做 baseline 自检（更快、更稳）
    model_name = "hfl/chinese-roberta-wwm-ext"
    tok = AutoTokenizer.from_pretrained(model_name)

    ds = CAILMultiTaskDataset(train_path, label_map_path, max_fact_chars=None)
    collate = Collator(tok, max_length=256)  # 先用 256 快速验证，后面再改 512/更长

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,   # Windows 建议 0，先别开多进程
        collate_fn=collate
    )

    batch = next(iter(loader))
    print("\nBatch keys:", list(batch.keys()))
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(k, v.shape, v.dtype)

    # 看一眼标签维度是不是符合你刚统计的类别数
    print("\nlabels_accusation dim:", batch["labels_accusation"].shape[-1])  # 期望 202
    print("labels_articles dim:", batch["labels_articles"].shape[-1])        # 期望 183


if __name__ == "__main__":
    main()
