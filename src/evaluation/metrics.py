"""
Multi-metric classification evaluation for CUB-200-2011 (Section 5.10).

Computes Top-1 Accuracy, Top-5 Accuracy, Macro & Weighted Precision / Recall / F1-score,
200x200 Confusion Matrix, per-species F1 rankings, and Top-K most confused species pairs,
with human-readable metric explanations and Markdown/JSON report generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None


METRIC_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "accuracy": {
        "name": "Top-1 Accuracy",
        "what_it_is": "Fraction of test/validation bird images whose highest-probability predicted species matches the ground-truth species.",
        "what_it_measures": "Overall exact species recognition across all 200 classes (random-chance baseline = 1/200 = 0.50%).",
    },
    "top5_accuracy": {
        "name": "Top-5 Accuracy",
        "what_it_is": "Fraction of images where the true bird species appears among the model's top 5 highest-scoring class predictions.",
        "what_it_measures": "Coarse-to-fine taxonomic neighborhood retrieval — whether the encoder maps visually similar subspecies/siblings (e.g., warblers, sparrows, terns) near the right region of latent space (random-chance = 2.50%).",
    },
    "macro_precision": {
        "name": "Macro Precision",
        "what_it_is": "Unweighted arithmetic mean of per-species Precision TP_c / (TP_c + FP_c) across active classes.",
        "what_it_measures": "How trustworthy positive predictions are across every bird species equally, penalizing false-alarm hub classes that attract many misclassifications.",
    },
    "macro_recall": {
        "name": "Macro Recall",
        "what_it_is": "Unweighted arithmetic mean of per-species Recall TP_c / (TP_c + FN_c) across present ground-truth classes.",
        "what_it_measures": "Class-balanced sensitivity — ensures rare or slightly smaller classes (29 vs. 30 training images) are not ignored.",
    },
    "macro_f1": {
        "name": "Macro F1-Score",
        "what_it_is": "Unweighted mean of per-species harmonic means 2 * (P_c * R_c) / (P_c + R_c).",
        "what_it_measures": "Balanced per-species classification quality; drops sharply if the model collapses onto a subset of easy classes even if overall accuracy looks moderate.",
    },
    "weighted_f1": {
        "name": "Weighted F1-Score",
        "what_it_is": "Support-weighted average of per-species F1-scores.",
        "what_it_measures": "Dataset-level harmonic balance of precision and recall weighted by each species' evaluation sample count.",
    },
    "confusion_matrix": {
        "name": "200x200 Confusion Matrix",
        "what_it_is": "Matrix C where C[i, j] counts birds of true species i predicted as species j.",
        "what_it_measures": "Diagonal dominance vs. off-diagonal inter-species confusion pairs (e.g., visually near-identical sibling species).",
    },
}


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = 200,
) -> np.ndarray:
    """
    Compute exact `(num_classes, num_classes)` integer confusion matrix where
    `cm[i, j]` is the number of samples with true label `i` predicted as `j`.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true shape {yt.shape} does not match y_pred shape {yp.shape}.")
    if yt.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    inferred_c = int(max(num_classes, int(yt.max()) + 1, int(yp.max()) + 1))
    valid = (yt >= 0) & (yt < inferred_c) & (yp >= 0) & (yp < inferred_c)
    flat_indices = yt[valid] * inferred_c + yp[valid]
    cm = np.bincount(flat_indices, minlength=inferred_c * inferred_c).reshape(inferred_c, inferred_c)
    return cm[:num_classes, :num_classes]


def compute_top_k_accuracy(
    logits: Union[np.ndarray, torch.Tensor],
    y_true: Union[Sequence[int], np.ndarray, torch.Tensor],
    k: int = 5,
) -> float:
    """Compute Top-K classification accuracy from `(N, C)` logits or probabilities."""
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits)
    else:
        logits_t = logits.detach().cpu()

    if isinstance(y_true, torch.Tensor):
        targets_t = y_true.detach().cpu().long().view(-1)
    else:
        targets_t = torch.as_tensor(np.asarray(y_true), dtype=torch.long).view(-1)

    if logits_t.numel() == 0 or targets_t.numel() == 0:
        return 0.0

    eff_k = min(max(1, int(k)), logits_t.size(1))
    _, topk_preds = logits_t.topk(eff_k, dim=1, largest=True, sorted=True)
    correct = topk_preds.eq(targets_t.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


def extract_top_confused_pairs(
    cm: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    top_k: int = 15,
) -> List[Dict[str, Any]]:
    """
    Extract the `top_k` largest off-diagonal entries `(true_class != pred_class)`
    from a confusion matrix.
    """
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    total_mistakes = int(off_diag.sum())
    if total_mistakes == 0:
        return []

    flat_order = np.argsort(off_diag.ravel())[::-1]
    row_sums = cm.sum(axis=1)
    pairs: List[Dict[str, Any]] = []

    for flat_idx in flat_order[:top_k]:
        cnt = int(off_diag.ravel()[flat_idx])
        if cnt <= 0:
            break
        true_idx, pred_idx = divmod(int(flat_idx), cm.shape[1])
        true_name = (
            class_names[true_idx]
            if class_names and true_idx < len(class_names)
            else f"Species_{true_idx:03d}"
        )
        pred_name = (
            class_names[pred_idx]
            if class_names and pred_idx < len(class_names)
            else f"Species_{pred_idx:03d}"
        )
        support = int(row_sums[true_idx])
        pairs.append(
            {
                "true_class_id": true_idx,
                "true_species": true_name,
                "pred_class_id": pred_idx,
                "pred_species": pred_name,
                "count": cnt,
                "true_class_support": support,
                "confusion_rate": round(float(cnt / max(support, 1)), 4),
            }
        )
    return pairs


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    logits: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_classes: int = 200,
    class_names: Optional[Sequence[str]] = None,
    top_confused_k: int = 15,
) -> Dict[str, Any]:
    """
    Compute a complete suite of multi-class classification metrics from ground-truth
    labels, predicted class indices, and optional `(N, C)` logits.
    """
    yt = np.asarray(y_true, dtype=np.int64).ravel()
    yp = np.asarray(y_pred, dtype=np.int64).ravel()
    n_samples = int(yt.size)

    if logits is not None:
        num_c_logits = int(logits.shape[1])
        num_classes = max(num_classes, num_c_logits)

    cm = compute_confusion_matrix(yt, yp, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_counts = cm.sum(axis=0).astype(np.float64)

    # Safe zero-division=0.0 per-class precision, recall, F1
    precision_per_class = np.divide(tp, pred_counts, out=np.zeros_like(tp), where=pred_counts > 0)
    recall_per_class = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision_per_class + recall_per_class
    f1_per_class = np.divide(
        2.0 * precision_per_class * recall_per_class,
        denom,
        out=np.zeros_like(tp),
        where=denom > 0,
    )

    # Macro averages computed across classes present in ground-truth (or predicted) so
    # small validation/unit-test subsets aren't diluted by absent classes
    present_mask = support > 0
    active_mask = present_mask if np.any(present_mask) else np.ones(num_classes, dtype=bool)

    macro_precision = float(precision_per_class[active_mask].mean()) if n_samples > 0 else 0.0
    macro_recall = float(recall_per_class[active_mask].mean()) if n_samples > 0 else 0.0
    macro_f1 = float(f1_per_class[active_mask].mean()) if n_samples > 0 else 0.0

    total_support = float(support.sum())
    if total_support > 0:
        weights = support / total_support
        weighted_precision = float(np.sum(precision_per_class * weights))
        weighted_recall = float(np.sum(recall_per_class * weights))
        weighted_f1 = float(np.sum(f1_per_class * weights))
        accuracy = float(tp.sum() / total_support)
    else:
        weighted_precision = weighted_recall = weighted_f1 = accuracy = 0.0

    top5_acc: Optional[float] = None
    if logits is not None and n_samples > 0:
        top5_acc = compute_top_k_accuracy(logits, yt, k=5)

    per_class_records: List[Dict[str, Any]] = []
    for cid in range(num_classes):
        if not active_mask[cid]:
            continue
        cname = (
            class_names[cid]
            if class_names and cid < len(class_names)
            else f"Species_{cid:03d}"
        )
        per_class_records.append(
            {
                "class_id": int(cid),
                "species": cname,
                "precision": round(float(precision_per_class[cid]), 4),
                "recall": round(float(recall_per_class[cid]), 4),
                "f1": round(float(f1_per_class[cid]), 4),
                "support": int(support[cid]),
            }
        )

    sorted_by_f1 = sorted(per_class_records, key=lambda r: (-r["f1"], -r["support"], r["class_id"]))
    confused_pairs = extract_top_confused_pairs(cm, class_names=class_names, top_k=top_confused_k)

    return {
        "num_samples": n_samples,
        "num_classes": num_classes,
        "num_present_classes": int(present_mask.sum()),
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_f1_values": [float(r["f1"]) for r in per_class_records],
        "best_10_classes": sorted_by_f1[:10],
        "worst_10_classes": sorted_by_f1[-10:][::-1],
        "top_confused_pairs": confused_pairs,
        "confusion_matrix": cm,
    }


def _interpret_classification_score(accuracy: float, macro_f1: float, num_classes: int = 200) -> str:
    chance = 1.0 / max(num_classes, 1)
    lift = accuracy / max(chance, 1e-8)
    gap = abs(accuracy - macro_f1)
    balance_note = (
        "Macro-F1 closely tracks Top-1 Accuracy, confirming balanced recognition across species rather than majority-class collapse."
        if gap < 0.05
        else "Noticeable gap between Top-1 Accuracy and Macro-F1 indicates uneven performance across hard vs. easy bird species."
    )
    return (
        f"Top-1 Accuracy = {accuracy * 100:.2f}% ({lift:.1f}x above uniform random chance of {chance * 100:.2f}% "
        f"on {num_classes} classes); Macro-F1 = {macro_f1 * 100:.2f}%. {balance_note}"
    )


def write_evaluation_report(
    metrics: Dict[str, Any],
    save_dir: Union[str, Path],
    model_name: str = "model",
    split_name: str = "val",
) -> Tuple[Path, Path]:
    """
    Save JSON metrics, `.npy` confusion matrix, and a Markdown interpretation report
    explaining what each metric is, what it measures, and what the model's score means.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = metrics.get("confusion_matrix")
    if isinstance(cm, np.ndarray):
        np.save(out_dir / f"confusion_matrix_{split_name}.npy", cm)

    serializable = {
        k: v
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }
    serializable["model_name"] = model_name
    serializable["split"] = split_name
    serializable["interpretation"] = _interpret_classification_score(
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        num_classes=int(metrics["num_classes"]),
    )
    serializable["metric_explanations"] = METRIC_EXPLANATIONS

    json_path = out_dir / f"metrics_{split_name}.json"
    json_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    top5_val = metrics.get("top5_accuracy")
    top5_str = f"{top5_val * 100:.2f}%" if top5_val is not None else "N/A"
    loss_val = metrics.get("loss")
    loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"

    md_lines = [
        f"# Classification Evaluation Report — `{model_name}` (`{split_name}` split)",
        "",
        f"- **Evaluated Samples**: `{metrics['num_samples']:,}` across `{metrics['num_present_classes']}` active species",
        f"- **Cross-Entropy Loss**: `{loss_str}`",
        f"- **Score Interpretation**: {serializable['interpretation']}",
        "",
        "## 1. Summary Metrics",
        "",
        "| Metric | Score | What It Measures |",
        "| :--- | :---: | :--- |",
        f"| **Top-1 Accuracy** | `{metrics['accuracy'] * 100:.2f}%` | {METRIC_EXPLANATIONS['accuracy']['what_it_measures']} |",
        f"| **Top-5 Accuracy** | `{top5_str}` | {METRIC_EXPLANATIONS['top5_accuracy']['what_it_measures']} |",
        f"| **Macro Precision** | `{metrics['macro_precision'] * 100:.2f}%` | {METRIC_EXPLANATIONS['macro_precision']['what_it_measures']} |",
        f"| **Macro Recall** | `{metrics['macro_recall'] * 100:.2f}%` | {METRIC_EXPLANATIONS['macro_recall']['what_it_measures']} |",
        f"| **Macro F1-Score** | `{metrics['macro_f1'] * 100:.2f}%` | {METRIC_EXPLANATIONS['macro_f1']['what_it_measures']} |",
        f"| **Weighted F1-Score** | `{metrics['weighted_f1'] * 100:.2f}%` | {METRIC_EXPLANATIONS['weighted_f1']['what_it_measures']} |",
        "",
        "## 2. Top Most-Confused Bird Species Pairs",
        "",
        "| True Species | Predicted As | Count | Confusion Rate |",
        "| :--- | :--- | :---: | :---: |",
    ]

    for pair in metrics.get("top_confused_pairs", [])[:12]:
        md_lines.append(
            f"| {pair['true_species']} | {pair['pred_species']} | `{pair['count']}` | `{pair['confusion_rate'] * 100:.1f}%` |"
        )
    if not metrics.get("top_confused_pairs"):
        md_lines.append("| *(No misclassifications)* | — | `0` | `0.0%` |")

    md_path = out_dir / f"evaluation_report_{split_name}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    criterion: Optional[nn.Module] = None,
    num_classes: int = 200,
    class_names: Optional[Sequence[str]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    model_name: str = "model",
    split_name: str = "val",
) -> Dict[str, Any]:
    """
    Evaluate a trained PyTorch classification model across `dataloader` and return
    loss + full multi-metric evaluation dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    running_loss = 0.0
    total = 0
    all_logits: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        bsz = inputs.size(0)
        running_loss += loss.item() * bsz
        total += bsz

        all_logits.append(outputs.detach().cpu())
        all_preds.append(outputs.argmax(dim=1).detach().cpu())
        all_labels.append(labels.detach().cpu())

    if total > 0:
        logits_cat = torch.cat(all_logits, dim=0)
        preds_cat = torch.cat(all_preds, dim=0).numpy()
        labels_cat = torch.cat(all_labels, dim=0).numpy()
        avg_loss = running_loss / total
    else:
        logits_cat = torch.empty((0, num_classes))
        preds_cat = np.empty((0,), dtype=np.int64)
        labels_cat = np.empty((0,), dtype=np.int64)
        avg_loss = 0.0

    results = compute_classification_metrics(
        y_true=labels_cat,
        y_pred=preds_cat,
        logits=logits_cat,
        num_classes=num_classes,
        class_names=class_names,
    )
    results["loss"] = float(avg_loss)

    if save_dir is not None:
        json_p, md_p = write_evaluation_report(
            results,
            save_dir=save_dir,
            model_name=model_name,
            split_name=split_name,
        )
        results["report_json_path"] = str(json_p)
        results["report_markdown_path"] = str(md_p)

    if wandb is not None and getattr(wandb, "run", None) is not None:
        log_payload = {
            f"{split_name}/eval_loss": results["loss"],
            f"{split_name}/accuracy": results["accuracy"],
            f"{split_name}/macro_precision": results["macro_precision"],
            f"{split_name}/macro_recall": results["macro_recall"],
            f"{split_name}/macro_f1": results["macro_f1"],
            f"{split_name}/weighted_f1": results["weighted_f1"],
        }
        if results.get("top5_accuracy") is not None:
            log_payload[f"{split_name}/top5_accuracy"] = results["top5_accuracy"]
        wandb.log(log_payload)

    return results

