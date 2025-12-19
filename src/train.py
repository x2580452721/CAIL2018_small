# -*- coding: utf-8 -*-
import json
import math
import time
from pathlib import Path
import torch
from torch.amp import autocast, GradScaler


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from src.data.dataset import CAILMultiTaskDataset, Collator
from src.models.multitask_head import MultiTaskClassifier


def find_data_root(project_root: Path) -> Path:
    """
    兼容两种结构：
      1) data/first_stage/...
      2) first_stage/...
    """
    a = project_root / "data" / "first_stage"
    b = project_root / "first_stage"
    if (a / "train_split.json").exists():
        return a
    if (b / "train_split.json").exists():
        return b
    raise FileNotFoundError("Cannot find first_stage folder in either data/first_stage or first_stage")


@torch.no_grad()
def eval_on_valid(model, loader, device, eval_batches=200, threshold=0.5):
    model.eval()
    bce_sigmoid = torch.sigmoid

    # multi-label micro-F1 统计
    acc_tp = acc_fp = acc_fn = 0
    art_tp = art_fp = art_fn = 0

    # term accuracy
    term_correct = 0
    term_total = 0

    it = iter(loader)
    for _ in range(eval_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        y_acc = batch["labels_accusation"].to(device)
        y_art = batch["labels_articles"].to(device)
        y_term = batch["labels_term"].to(device)

        acc_logits, art_logits, term_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # accusation micro-F1
        acc_pred = (bce_sigmoid(acc_logits) >= threshold).to(torch.int32)
        acc_true = (y_acc >= 0.5).to(torch.int32)
        acc_tp += int(((acc_pred == 1) & (acc_true == 1)).sum().item())
        acc_fp += int(((acc_pred == 1) & (acc_true == 0)).sum().item())
        acc_fn += int(((acc_pred == 0) & (acc_true == 1)).sum().item())

        # article micro-F1
        art_pred = (bce_sigmoid(art_logits) >= threshold).to(torch.int32)
        art_true = (y_art >= 0.5).to(torch.int32)
        art_tp += int(((art_pred == 1) & (art_true == 1)).sum().item())
        art_fp += int(((art_pred == 1) & (art_true == 0)).sum().item())
        art_fn += int(((art_pred == 0) & (art_true == 1)).sum().item())

        # term accuracy
        term_hat = term_logits.argmax(dim=-1)
        term_correct += int((term_hat == y_term).sum().item())
        term_total += int(y_term.numel())

    def micro_f1(tp, fp, fn):
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return (2 * precision * recall) / (precision + recall + 1e-12)

    return {
        "accusation_micro_f1": micro_f1(acc_tp, acc_fp, acc_fn),
        "article_micro_f1": micro_f1(art_tp, art_fp, art_fn),
        "term_acc": (term_correct / max(1, term_total)),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = find_data_root(project_root)

    train_path = data_root / "train_split.json"
    valid_path = data_root / "valid_split.json"
    label_map_path = data_root / "label_map.json"

    out_dir = project_root / "outputs" / "exp_roberta_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- 可调超参（先给一套能跑通的稳配置）---------
    model_name = "hfl/chinese-roberta-wwm-ext"
    max_len = 256          # 先用256稳定跑通；显存够再改512
    batch_size = 4        # 显存不够就改 8/4
    lr = 2e-5
    epochs = 1             # 先跑1轮看指标，后面再加
    warmup_ratio = 0.01
    log_every = 50
    eval_every = 500       # 每500 step 验证一次（可改大一点）
    eval_batches = 50     # 验证时只跑前200个batch，省时间（想更准就调大）
    grad_accum = 4         # 显存不够就用2/4做梯度累积
    loss_w_acc = 1.0
    loss_w_art = 1.0
    loss_w_term = 1.0
    threshold = 0.25
    # ------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("data_root:", data_root)
    print("out_dir:", out_dir)

    # 读取类别数
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    num_acc = len(lm["accusation2id"])
    num_art = len(lm["article2id"])
    num_term = len(lm.get("term_bins", [])) or 9
    print("num_acc:", num_acc, "num_art:", num_art, "num_term:", num_term)

    # tokenizer + encoder
    tok = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size = encoder.config.hidden_size

    model = MultiTaskClassifier(encoder, hidden_size, num_acc=num_acc, num_art=num_art, num_term=num_term, dropout=0.1)
    model.to(device)

    # datasets
    train_ds = CAILMultiTaskDataset(train_path, label_map_path, max_fact_chars=None)
    valid_ds = CAILMultiTaskDataset(valid_path, label_map_path, max_fact_chars=None)

    collate = Collator(tok, max_length=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate
    )

    # losses
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))


    # 保存训练配置
    cfg = {
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "grad_accum": grad_accum,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "threshold": threshold,
        "loss_weights": {"acc": loss_w_acc, "art": loss_w_art, "term": loss_w_term},
        "num_labels": {"acc": num_acc, "art": num_art, "term": num_term},
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -1.0
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print("Start training...")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            y_acc = batch["labels_accusation"].to(device)
            y_art = batch["labels_articles"].to(device)
            y_term = batch["labels_term"].to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                acc_logits, art_logits, term_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss_acc = bce(acc_logits, y_acc)
                loss_art = bce(art_logits, y_art)
                loss_term = ce(term_logits, y_term)
                loss = loss_w_acc * loss_acc + loss_w_art * loss_art + loss_w_term * loss_term
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)  # 如果overflow，这一步会跳过optimizer.step
                scaler.update()
                scale_after = scaler.get_scale()

                # 只有没overflow（scale没下降）才推进scheduler
                if scale_after >= scale_before:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += float(loss.item()) * grad_accum

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[ep {ep}] step {global_step}/{total_steps} "
                          f"loss={running_loss/log_every:.4f} "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}"
                          f"time={elapsed/60:.1f}m")
                    running_loss = 0.0

                if global_step % eval_every == 0:
                    metrics = eval_on_valid(model, valid_loader, device, eval_batches=eval_batches, threshold=threshold)
                    score = (metrics["accusation_micro_f1"] + metrics["article_micro_f1"] + metrics["term_acc"]) / 3.0
                    print(f"  [valid] accF1={metrics['accusation_micro_f1']:.4f} "
                          f"artF1={metrics['article_micro_f1']:.4f} "
                          f"termAcc={metrics['term_acc']:.4f} "
                          f"score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        save_dir = out_dir / "best"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        # 保存 encoder + tokenizer（方便复现）
                        model.encoder.save_pretrained(save_dir / "encoder")
                        tok.save_pretrained(save_dir / "tokenizer")

                        # 保存多任务头参数
                        torch.save(model.head.state_dict(), save_dir / "multitask_head.pt")

                        (save_dir / "best_metrics.json").write_text(
                            json.dumps({"best_score": best_score, **metrics}, ensure_ascii=False, indent=2),
                            encoding="utf-8"
                        )
                        print("  saved best to:", save_dir)

        # 每个 epoch 结束再评一次
        metrics = eval_on_valid(model, valid_loader, device, eval_batches=eval_batches, threshold=threshold)
        score = (metrics["accusation_micro_f1"] + metrics["article_micro_f1"] + metrics["term_acc"]) / 3.0
        print(f"[epoch end {ep}] accF1={metrics['accusation_micro_f1']:.4f} "
              f"artF1={metrics['article_micro_f1']:.4f} termAcc={metrics['term_acc']:.4f} score={score:.4f}")

    print("Done. best_score=", best_score)
    print("outputs in:", out_dir)


if __name__ == "__main__":
    main()
