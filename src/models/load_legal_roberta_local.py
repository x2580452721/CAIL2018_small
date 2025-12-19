from pathlib import Path
from transformers import AutoTokenizer, AutoModel

root = Path(__file__).resolve().parents[2]
model_dir = root / "pretrained" / "LegalRoBERTa"  # 你移动后的目录

# tokenizer 用通用中文BERT词表（最稳）
tok = AutoTokenizer.from_pretrained("bert-base-chinese")

# 模型权重用你本地的 LegalRoBERTa
enc = AutoModel.from_pretrained(model_dir)

print("OK tokenizer:", tok.__class__.__name__)
print("OK model:", enc.__class__.__name__)
