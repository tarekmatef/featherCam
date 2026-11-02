"""
Matplotlib visualizations for training curves, multi-metric evaluation,
200x200 confusion matrices, top-confused bird species pairs, and model comparisons.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _prepare_save_path(save_path: Union[str, Path]) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(
    history: Dict[str, Sequence[float]],
    save_path: Union[str, Path],
    title: str = "Training vs. Validation Curves",
) -> Path:
    """
    Plot epoch-wise Training vs. Validation Loss and Accuracy curves and highlight
    the best validation epoch.
    """
    out_path = _prepare_save_path(save_path)
    tr_loss = list(history.get("train_loss", []))
    va_loss = list(history.get("val_loss", []))
    tr_acc = [float(a) * 100.0 for a in history.get("train_acc", [])]
    va_acc = [float(a) * 100.0 for a in history.get("val_acc", [])]

    epochs_tr = np.arange(1, len(tr_loss) + 1)
    epochs_va = np.arange(1, len(va_loss) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    if len(tr_loss) > 0:
        ax_loss.plot(epochs_tr, tr_loss, marker="o", markersize=4, linewidth=2.0, color="#2b6cb0", label="Train Loss")
    if len(va_loss) > 0:
        ax_loss.plot(epochs_va, va_loss, marker="s", markersize=4, linewidth=2.0, color="#dd6b20", label="Val Loss")
        best_idx = int(np.argmin(va_loss))
        ax_loss.axvline(best_idx + 1, color="#2f855a", linestyle="--", alpha=0.8, label=f"Best Val Loss (Ep {best_idx + 1}: {va_loss[best_idx]:.3f})")

    ax_loss.set_title("Cross-Entropy Loss Curve", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    if tr_loss or va_loss:
        ax_loss.legend(fontsize=8.5)

    if len(tr_acc) > 0:
        ax_acc.plot(epochs_tr, tr_acc, marker="o", markersize=4, linewidth=2.0, color="#2b6cb0", label="Train Top-1 Acc")
    if len(va_acc) > 0:
        ax_acc.plot(epochs_va, va_acc, marker="s", markersize=4, linewidth=2.0, color="#dd6b20", label="Val Top-1 Acc")
        best_acc_idx = int(np.argmax(va_acc))
        ax_acc.scatter([best_acc_idx + 1], [va_acc[best_acc_idx]], color="#2f855a", s=70, zorder=5, label=f"Peak Val Acc (Ep {best_acc_idx + 1}: {va_acc[best_acc_idx]:.2f}%)")

    ax_acc.set_title("Top-1 Classification Accuracy (%)", fontweight="bold")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(alpha=0.3)
    if tr_acc or va_acc:
        ax_acc.legend(fontsize=8.5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    max_zoom_classes: int = 18,
    title: str = "Classification Confusion Matrix Analysis",
) -> Path:
    """
    Plot a 2-panel confusion matrix diagnostic:
      - Left: Global row-normalized `(C x C)` confusion matrix across all species.
      - Right: Annotated sub-matrix of the most-confused species (highest off-diagonal mass).
    """
    out_path = _prepare_save_path(save_path)
    cm_arr = np.asarray(cm, dtype=np.float64)
    n_classes = cm_arr.shape[0]

    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums > 0)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6.2))

    im0 = ax_full.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=max(float(cm_norm.max()), 1e-3), aspect="auto")
    fig.colorbar(im0, ax=ax_full, fraction=0.046, pad=0.04, label="Row-Normalized Recall / Confusion Rate")
    ax_full.set_title(f"Global Normalized Confusion Matrix ({n_classes}x{n_classes})", fontweight="bold")
    ax_full.set_xlabel("Predicted Class ID")
    ax_full.set_ylabel("True Class ID")

    # Select subset of classes involved in the highest off-diagonal errors
    off_diag = cm_arr.copy()
    np.fill_diagonal(off_diag, 0.0)
    class_error_mass = off_diag.sum(axis=1) + off_diag.sum(axis=0)
    k = min(max_zoom_classes, n_classes)
    if class_error_mass.sum() > 0:
        zoom_ids = np.sort(np.argsort(-class_error_mass)[:k])
    else:
        present = np.where(cm_arr.sum(axis=1) > 0)[0]
        zoom_ids = present[:k] if len(present) > 0 else np.arange(k)

    sub_cm = cm_arr[np.ix_(zoom_ids, zoom_ids)].astype(np.int64)
    im1 = ax_zoom.imshow(sub_cm, cmap="OrRd", aspect="auto")
    fig.colorbar(im1, ax=ax_zoom, fraction=0.046, pad=0.04, label="Sample Count")

    labels = [
        (class_names[i][:20] if class_names and i < len(class_names) else f"Cls {i}")
        for i in zoom_ids
    ]
    ax_zoom.set_xticks(np.arange(len(zoom_ids)))
    ax_zoom.set_yticks(np.arange(len(zoom_ids)))
    ax_zoom.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax_zoom.set_yticklabels(labels, fontsize=7.5)
    ax_zoom.set_xlabel("Predicted Species")
    ax_zoom.set_ylabel("True Species")
    ax_zoom.set_title(f"Most-Confused Species Sub-Matrix ({len(zoom_ids)}x{len(zoom_ids)})", fontweight="bold")

    if len(zoom_ids) <= 20:
        thresh = sub_cm.max() * 0.5 if sub_cm.size > 0 else 0
        for r in range(len(zoom_ids)):
            for c in range(len(zoom_ids)):
                val = int(sub_cm[r, c])
                if val > 0:
                    ax_zoom.text(
                        c,
                        r,
                        str(val),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > thresh else "#1a202c",
                    )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_evaluation_diagnostics(
    metrics: Dict[str, Any],
    save_path: Union[str, Path],
    top_k_pairs: int = 12,
    title: str = "Per-Species F1 Distribution & Top Inter-Species Confusions",
) -> Path:
    """
    Plot a 2-panel classification diagnostic:
      - Left: Top-K most confused `(True Species -> Predicted Species)` pairs.
      - Right: Per-species F1-score distribution across all active classes.
    """
    out_path = _prepare_save_path(save_path)
    fig, (ax_pairs, ax_f1) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    pairs = list(metrics.get("top_confused_pairs", []))[:top_k_pairs][::-1]
    if pairs:
        pair_labels = [f"{p['true_species'][:18]} -> {p['pred_species'][:18]}" for p in pairs]
        pair_counts = [int(p["count"]) for p in pairs]
        y_pos = np.arange(len(pairs))
        bars = ax_pairs.barh(y_pos, pair_counts, color="#e53e3e", alpha=0.85, height=0.72)
        ax_pairs.set_yticks(y_pos)
        ax_pairs.set_yticklabels(pair_labels, fontsize=8.5)
        for bar, cnt in zip(bars, pair_counts):
            ax_pairs.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(cnt), va="center", fontsize=8)
    else:
        ax_pairs.text(0.5, 0.5, "No off-diagonal misclassifications!", ha="center", va="center", fontsize=10)
    ax_pairs.set_title("Top Confused Bird Species Pairs", fontweight="bold")
    ax_pairs.set_xlabel("Misclassified Sample Count")
    ax_pairs.grid(axis="x", alpha=0.25)

    f1_vals = np.asarray(metrics.get("per_class_f1_values", [0.0]), dtype=np.float64) * 100.0
    ax_f1.hist(f1_vals, bins=20, range=(0.0, 100.0), color="#2b6cb0", alpha=0.82, edgecolor="white")
    macro_f1 = float(metrics.get("macro_f1", 0.0)) * 100.0
    acc = float(metrics.get("accuracy", 0.0)) * 100.0
    ax_f1.axvline(macro_f1, color="#dd6b20", linestyle="--", linewidth=1.6, label=f"Macro F1 = {macro_f1:.2f}%")
    ax_f1.axvline(acc, color="#2f855a", linestyle="-.", linewidth=1.6, label=f"Top-1 Acc = {acc:.2f}%")
    ax_f1.set_title("Per-Species F1-Score Distribution (%)", fontweight="bold")
    ax_f1.set_xlabel("Species F1-Score (%)")
    ax_f1.set_ylabel("Number of Species")
    ax_f1.legend(fontsize=8.5)
    ax_f1.grid(alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    save_path: Union[str, Path],
    metric_keys: Sequence[str] = ("accuracy", "top5_accuracy", "macro_f1", "macro_precision", "macro_recall"),
    title: str = "Model & Experiment Comparison (From-Scratch vs. Transfer / Tuning)",
) -> Path:
    """
    Plot a grouped bar chart comparing multiple models, architectures, or hyperparameter
    search methods across multiple evaluation metrics.

    Args:
        model_results: Mapping `{model_label: {metric_name: score_in_0_to_1}}`.
        save_path: Destination PNG path.
        metric_keys: Ordered metrics to display per model.
    """
    out_path = _prepare_save_path(save_path)
    model_names = list(model_results.keys())
    active_metrics = [
        m for m in metric_keys
        if any(model_results[mn].get(m) is not None for mn in model_names)
    ]
    if not model_names or not active_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No comparison metrics available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    n_models = len(model_names)
    n_metrics = len(active_metrics)
    x = np.arange(n_metrics)
    total_width = 0.76
    bar_w = total_width / max(n_models, 1)
    palette = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030", "#319795"]

    fig, ax = plt.subplots(figsize=(max(9.0, 2.0 * n_metrics), 5.2))
    for m_idx, m_name in enumerate(model_names):
        vals = [float(model_results[m_name].get(k, 0.0) or 0.0) * 100.0 for k in active_metrics]
        offsets = x - total_width / 2 + (m_idx + 0.5) * bar_w
        bars = ax.bar(
            offsets,
            vals,
            width=bar_w,
            label=m_name,
            color=palette[m_idx % len(palette)],
            edgecolor="white",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    pretty_names = {
        "accuracy": "Top-1 Acc",
        "top5_accuracy": "Top-5 Acc",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "macro_precision": "Macro Prec",
        "macro_recall": "Macro Recall",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_names.get(k, k) for k in active_metrics], fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path

