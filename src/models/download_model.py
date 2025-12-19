# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel

def main():
    name = "thunlp/Lawformer"
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name)
    print("downloaded & cached:", name)
    print("tokenizer vocab size:", tok.vocab_size)
    print("model type:", enc.__class__.__name__)

if __name__ == "__main__":
    main()
