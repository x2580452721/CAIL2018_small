# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """
    输入：encoder 的 CLS 向量
    输出：
      - accusation_logits: [B, num_acc]  (多标签)
      - article_logits:    [B, num_art]  (多标签)
      - term_logits:       [B, num_term] (单标签多分类)
    """
    def __init__(self, hidden_size: int, num_acc: int, num_art: int, num_term: int = 9, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.acc_head = nn.Linear(hidden_size, num_acc)
        self.art_head = nn.Linear(hidden_size, num_art)
        self.term_head = nn.Linear(hidden_size, num_term)

    def forward(self, cls_vec: torch.Tensor):
        x = self.dropout(cls_vec)
        return self.acc_head(x), self.art_head(x), self.term_head(x)


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_acc: int, num_art: int, num_term: int = 9, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = MultiTaskHead(hidden_size, num_acc, num_art, num_term=num_term, dropout=dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """
        ✅ 关键改动：支持 Longformer 的 global_attention_mask 等额外参数
        RoBERTa 没有这些参数也不影响（kwargs 为空即可）
        """
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        cls_vec = out.last_hidden_state[:, 0]
        return self.head(cls_vec)
