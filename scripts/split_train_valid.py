import json, random
from pathlib import Path

src = Path(r"D:\pycharm\CAIL2018_small\first_stage\train.json")
out_train = src.parent / "train_split.json"
out_valid = src.parent / "valid_split.json"

random.seed(42)
valid_ratio = 0.1

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(line)

random.shuffle(rows)
n_valid = int(len(rows) * valid_ratio)
valid_rows = rows[:n_valid]
train_rows = rows[n_valid:]

out_train.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
out_valid.write_text("\n".join(valid_rows) + "\n", encoding="utf-8")

print("total:", len(rows), "train:", len(train_rows), "valid:", len(valid_rows))
print("saved:", out_train, out_valid)
