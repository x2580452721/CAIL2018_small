# -*- coding: utf-8 -*-
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.data.dataset import CAILMultiTaskDataset, Collator
from src.models.multitask_head import MultiTaskClassifier


def find_data_root(project_root: Path) -> Path:
    a = project_root / "data" / "first_stage"
    b = project_root / "first_stage"
    if (a / "test.json").exists():
        return a
    if (b / "test.json").exists():
        return b
    raise FileNotFoundError("Cannot find first_stage folder (data/first_stage or first_stage)")


@torch.no_grad()
def eval_and_predict(model, loader, device, label_map, out_path: Path,
                     threshold=0.20, max_batches=None,
                     log_every=200, buffer_lines=2000):
    model.eval()
    id2acc = {v: k for k, v in label_map["accusation2id"].items()}
    id2art = {v: k for k, v in label_map["article2id"].items()}

    acc_tp = acc_fp = acc_fn = 0
    art_tp = art_fp = art_fn = 0
    term_correct = 0
    term_total = 0
    has_labels = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open("w", encoding="utf-8")

    buf = []
    total_batches = len(loader)

    for bi, batch in enumerate(loader, start=1):
        if max_batches is not None and bi > max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        y_acc = batch.get("labels_accusation")
        y_art = batch.get("labels_articles")
        y_term = batch.get("labels_term")

        acc_logits, art_logits, term_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 写预测（缓冲）
        acc_prob = torch.sigmoid(acc_logits).detach().cpu()
        art_prob = torch.sigmoid(art_logits).detach().cpu()
        term_pred = term_logits.argmax(dim=-1).detach().cpu()

        bs = acc_prob.size(0)
        for i in range(bs):
            acc_ids = (acc_prob[i] >= threshold).nonzero(as_tuple=True)[0].tolist()
            art_ids = (art_prob[i] >= threshold).nonzero(as_tuple=True)[0].tolist()

            acc_sorted = sorted([(id2acc[j], float(acc_prob[i, j])) for j in acc_ids],
                                key=lambda x: x[1], reverse=True)[:10]
            art_sorted = sorted([(id2art[j], float(art_prob[i, j])) for j in art_ids],
                                key=lambda x: x[1], reverse=True)[:10]

            rec = {
                "pred_accusation_top": acc_sorted,
                "pred_articles_top": art_sorted,
                "pred_term_bin": int(term_pred[i].item()),
            }
            buf.append(json.dumps(rec, ensure_ascii=False))

        if len(buf) >= buffer_lines:
            f.write("\n".join(buf) + "\n")
            buf.clear()
            f.flush()

        # 有标签才算指标（test 若没标签，这段会一直不进）
        if (y_acc is not None) and (y_art is not None) and (y_term is not None):
            has_labels = True
            y_acc = y_acc.to(device)
            y_art = y_art.to(device)
            y_term = y_term.to(device)

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

            term_hat = term_logits.argmax(dim=-1)
            term_correct += int((term_hat == y_term).sum().item())
            term_total += int(y_term.numel())

        # 进度日志
        if (bi == 1) or (bi % log_every == 0) or (bi == total_batches):
            print(f"[test] batch {bi}/{total_batches} (writing preds...)")

    # 把剩余缓冲写完
    if buf:
        f.write("\n".join(buf) + "\n")
        buf.clear()
        f.flush()

    f.close()

    def micro_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        return (2 * p * r) / (p + r + 1e-12)

    metrics = {"saved_pred": str(out_path), "threshold": threshold}
    if has_labels:
        acc_f1 = micro_f1(acc_tp, acc_fp, acc_fn)
        art_f1 = micro_f1(art_tp, art_fp, art_fn)
        term_acc = term_correct / max(1, term_total)
        score = (acc_f1 + art_f1 + term_acc) / 3.0
        metrics.update({
            "accusation_micro_f1": acc_f1,
            "article_micro_f1": art_f1,
            "term_acc": term_acc,
            "score": score,
        })
    else:
        metrics["note"] = "test set has no labels; only predictions were saved."

    return metrics



def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = find_data_root(project_root)

    # ====== 选择你保存的 best 模型 ======
    best_dir = project_root / "outputs" / "exp_lawformer" / "best"
    encoder_dir = best_dir / "encoder"
    tok_dir = best_dir / "tokenizer"
    head_path = best_dir / "multitask_head.pt"

    if not (encoder_dir.exists() and tok_dir.exists() and head_path.exists()):
        raise FileNotFoundError(f"best model not found under: {best_dir}")

    label_map_path = data_root / "label_map.json"
    test_path = data_root / "test.json"
    out_pred = project_root / "outputs" / "exp_lawformer" / "test_pred.jsonl"

    threshold = 0.20
    max_len = 256
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("data_root:", data_root)
    print("test_path:", test_path)
    print("best_dir:", best_dir)

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_acc = len(label_map["accusation2id"])
    num_art = len(label_map["article2id"])
    num_term = len(label_map.get("term_bins", [])) or 9

    tok = AutoTokenizer.from_pretrained(tok_dir)
    encoder = AutoModel.from_pretrained(encoder_dir)

    model = MultiTaskClassifier(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        num_acc=num_acc,
        num_art=num_art,
        num_term=num_term,
        dropout=0.1,
    )

    state = torch.load(head_path, map_location="cpu")
    model.head.load_state_dict(state)

    model.to(device)

    ds = CAILMultiTaskDataset(test_path, label_map_path, max_fact_chars=None)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=Collator(tok, max_length=max_len),
    )

    metrics = eval_and_predict(model, loader, device, label_map, out_pred, threshold=threshold, max_batches=None)
    print("metrics:", metrics)

    # 保存 metrics
    (project_root / "outputs" / "exp_roberta_baseline" / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("saved test metrics to:", project_root / "outputs" / "exp_roberta_baseline" / "test_metrics.json")


if __name__ == "__main__":
    main()
