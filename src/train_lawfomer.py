# -*- coding: utf-8 -*-
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup


from src.data.dataset import CAILMultiTaskDataset, Collator
from src.models.multitask_head import MultiTaskClassifier


def find_data_root(project_root: Path) -> Path:
    a = project_root / "data" / "first_stage"
    b = project_root / "first_stage"
    if (a / "train_split.json").exists():
        return a
    if (b / "train_split.json").exists():
        return b
    raise FileNotFoundError("Cannot find first_stage folder in either data/first_stage or first_stage")


@torch.no_grad()
def eval_on_valid(model, loader, device, eval_batches=100, threshold=0.20):
    model.eval()

    acc_tp = acc_fp = acc_fn = 0
    art_tp = art_fp = art_fn = 0
    term_correct = 0
    term_total = 0

    it = iter(loader)
    for _ in range(eval_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        # inputs
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        global_attention_mask = batch.get("global_attention_mask")
        if global_attention_mask is not None:
            global_attention_mask = global_attention_mask.to(device)

        # labels
        y_acc = batch["labels_accusation"].to(device)
        y_art = batch["labels_articles"].to(device)
        y_term = batch["labels_term"].to(device)

        acc_logits, art_logits, term_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            global_attention_mask=global_attention_mask,
        )

        # micro-F1 (multi-label)
        acc_pred = (torch.sigmoid(acc_logits) >= threshold).to(torch.int32)
        acc_true = (y_acc >= 0.5).to(torch.int32)
        acc_tp += int(((acc_pred == 1) & (acc_true == 1)).sum().item())
        acc_fp += int(((acc_pred == 1) & (acc_true == 0)).sum().item())
        acc_fn += int(((acc_pred == 0) & (acc_true == 1)).sum().item())

        art_pred = (torch.sigmoid(art_logits) >= threshold).to(torch.int32)
        art_true = (y_art >= 0.5).to(torch.int32)
        art_tp += int(((art_pred == 1) & (art_true == 1)).sum().item())
        art_fp += int(((art_pred == 1) & (art_true == 0)).sum().item())
        art_fn += int(((art_pred == 0) & (art_true == 1)).sum().item())

        # term acc
        term_hat = term_logits.argmax(dim=-1)
        term_correct += int((term_hat == y_term).sum().item())
        term_total += int(y_term.numel())

    def micro_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        return (2 * p * r) / (p + r + 1e-12)

    metrics = {
        "accusation_micro_f1": micro_f1(acc_tp, acc_fp, acc_fn),
        "article_micro_f1": micro_f1(art_tp, art_fp, art_fn),
        "term_acc": term_correct / max(1, term_total),
    }
    metrics["score"] = (metrics["accusation_micro_f1"] + metrics["article_micro_f1"] + metrics["term_acc"]) / 3.0
    return metrics


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = find_data_root(project_root)

    train_path = data_root / "train_split.json"
    valid_path = data_root / "valid_split.json"
    label_map_path = data_root / "label_map.json"

    # ====== 你改成 lawformer 时，建议也改 out_dir 名字 ======
    model_name = "thunlp/Lawformer"
    out_dir = project_root / "outputs" / "exp_lawformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- 超参（Lawformer 建议更保守）---------
    max_len = 256
    batch_size = 4          # 8GB 显存建议 1~2
    grad_accum = 8          # 等效 batch=8
    lr = 2e-5
    epochs = 1
    warmup_ratio = 0.01
    log_every = 50
    eval_every = 500
    eval_batches = 50
    threshold = 0.20

    loss_w_acc = 1.0
    loss_w_art = 1.0
    loss_w_term = 1.0
    # --------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("data_root:", data_root)
    print("out_dir:", out_dir)

    lm = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_acc = len(lm["accusation2id"])
    num_art = len(lm["article2id"])
    num_term = len(lm.get("term_bins", [])) or 9
    print("num_acc:", num_acc, "num_art:", num_art, "num_term:", num_term)

    tok = AutoTokenizer.from_pretrained(model_name)

    # ✅ 先改 config，再构造模型，才能真正生效到每一层 self-attention
    config = AutoConfig.from_pretrained(model_name)
    is_longformer = getattr(config, "model_type", "") == "longformer"
    if is_longformer:
        config.attention_window = [256] * config.num_hidden_layers
        print("set longformer attention_window -> 256 (before model load)")

    encoder = AutoModel.from_pretrained(model_name, config=config)

    model = MultiTaskClassifier(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        num_acc=num_acc,
        num_art=num_art,
        num_term=num_term,
        dropout=0.1,
    ).to(device)

    train_ds = CAILMultiTaskDataset(train_path, label_map_path)
    valid_ds = CAILMultiTaskDataset(valid_path, label_map_path)

    collate = Collator(
        tokenizer=tok,
        max_length=max_len,
        add_global_attention_mask=is_longformer,   # ✅ Longformer 才加
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    cfg = {
        "model_name": model_name,
        "max_len": max_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "epochs": epochs,
        "threshold": threshold,
        "is_longformer": is_longformer,
        "attention_window": 256 if is_longformer else None,
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
            global_attention_mask = batch.get("global_attention_mask")
            if global_attention_mask is not None:
                global_attention_mask = global_attention_mask.to(device)

            y_acc = batch["labels_accusation"].to(device)
            y_art = batch["labels_articles"].to(device)
            y_term = batch["labels_term"].to(device)

            with autocast("cuda", enabled=(device.type == "cuda")):
                acc_logits, art_logits, term_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    global_attention_mask=global_attention_mask,
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
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_after >= scale_before:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                running_loss += float(loss.item()) * grad_accum

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[ep {ep}] step {global_step}/{total_steps} "
                          f"loss={running_loss/log_every:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} "
                          f"time={elapsed/60:.1f}m")
                    running_loss = 0.0

                if global_step % eval_every == 0:
                    metrics = eval_on_valid(model, valid_loader, device, eval_batches=eval_batches, threshold=threshold)
                    print(f"  [valid] accF1={metrics['accusation_micro_f1']:.4f} "
                          f"artF1={metrics['article_micro_f1']:.4f} "
                          f"termAcc={metrics['term_acc']:.4f} score={metrics['score']:.4f}")

                    if metrics["score"] > best_score:
                        best_score = metrics["score"]
                        save_dir = out_dir / "best"
                        save_dir.mkdir(parents=True, exist_ok=True)

                        model.encoder.save_pretrained(save_dir / "encoder")
                        tok.save_pretrained(save_dir / "tokenizer")
                        torch.save(model.head.state_dict(), save_dir / "multitask_head.pt")

                        (save_dir / "best_metrics.json").write_text(
                            json.dumps({"best_score": best_score, "threshold": threshold, **metrics},
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8"
                        )
                        print("  saved best to:", save_dir)

    print("Done. best_score=", best_score)
    print("outputs in:", out_dir)


if __name__ == "__main__":
    main()
