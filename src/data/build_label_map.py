# -*- coding: utf-8 -*-
import json
from pathlib import Path
import argparse


def build_label_map(train_jsonl: Path, out_json: Path):
    acc_set = set()
    art_set = set()

    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("meta", {})

            for a in meta.get("accusation", []):
                acc_set.add(str(a))
            for r in meta.get("relevant_articles", []):
                art_set.add(str(r))

    acc_list = sorted(acc_set)

    def art_key(x: str):
        return int(x) if x.isdigit() else x

    art_list = sorted(art_set, key=art_key)

    label_map = {
        "accusation2id": {a: i for i, a in enumerate(acc_list)},
        "id2accusation": acc_list,
        "article2id": {a: i for i, a in enumerate(art_list)},
        "id2article": art_list,
        "term_bins": ["death", "life", "0", "1-6", "7-12", "13-36", "37-60", "61-120", ">120"],
    }

    out_json.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_json)
    print("num_accusation:", len(acc_list), "num_article:", len(art_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=str(Path(r"D:\pycharm\CAIL2018_small\first_stage\train_split.json")))
    ap.add_argument("--out", type=str, default=str(Path(r"D:\pycharm\CAIL2018_small\first_stage/label_map.json")))
    args = ap.parse_args()

    train_path = Path(args.train)
    out_path = Path(args.out)

    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    build_label_map(train_path, out_path)


if __name__ == "__main__":
    main()
