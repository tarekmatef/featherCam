"""
Matplotlib visualizations for Task 1.1: Data Understanding, EDA, Caption Analysis,
and Augmentation Strategy.
"""

from pathlib import Path
import textwrap
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from src.data.cub200 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        denormalize_tensor,
        get_cub200_transforms,
    )
except ImportError:
    from data.cub200 import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        denormalize_tensor,
        get_cub200_transforms,
    )


def _prepare_save_path(save_path: Union[str, Path]) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_sample_images(
    samples: Sequence[Dict[str, Any]],
    save_path: Union[str, Path],
    n_cols: int = 4,
    title: str = "CUB-200-2011 Sample Birds & Cleaned Captions",
) -> Path:
    """
    Plot a grid of sample bird photographs with species labels, original image
    dimensions, and a cleaned natural-language caption.

    Each item in `samples` expects keys:
      - `'image'`: PIL.Image or `(H, W, 3)` RGB array
      - `'class_name'`: Species name string
      - `'caption'`: Optional cleaned caption string
    """
    out_path = _prepare_save_path(save_path)
    n = len(samples)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sample images provided", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    cols = min(max(1, n_cols), n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.3 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx >= n:
            ax.axis("off")
            continue

        item = samples[idx]
        img = item["image"]
        if isinstance(img, Image.Image):
            w, h = img.size
            arr = np.asarray(img.convert("RGB"))
        else:
            arr = np.asarray(img)
            h, w = arr.shape[:2]

        cls_name = str(item.get("class_name", "Unknown Species"))
        caption = item.get("caption")

        ax.imshow(arr)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{cls_name}\n({w}x{h} px)", fontsize=9.5, fontweight="bold", pad=5)

        if caption:
            wrapped = "\n".join(textwrap.wrap(f'"{caption}"', width=34)[:3])
            ax.set_xlabel(wrapped, fontsize=8, style="italic", labelpad=5)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_class_distribution(
    train_counts: Dict[int, int],
    save_path: Union[str, Path],
    val_counts: Optional[Dict[int, int]] = None,
    test_counts: Optional[Dict[int, int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Plot per-species sample distribution across Train / Val / Test splits and
    highlight classes with the lowest and highest sample counts.
    """
    out_path = _prepare_save_path(save_path)
    all_classes = sorted(
        set(train_counts.keys())
        | set((val_counts or {}).keys())
        | set((test_counts or {}).keys())
    )
    if not all_classes:
        all_classes = [0]
        train_counts = {0: 0}

    tr_vals = np.array([train_counts.get(c, 0) for c in all_classes], dtype=np.int32)
    va_vals = np.array([(val_counts or {}).get(c, 0) for c in all_classes], dtype=np.int32)
    te_vals = np.array([(test_counts or {}).get(c, 0) for c in all_classes], dtype=np.int32)
    totals = tr_vals + va_vals + te_vals

    order = np.argsort(-totals)
    x = np.arange(len(all_classes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.3, 1.0]})

    # Panel 1: Stacked distribution across all classes sorted by total count
    ax1.bar(x, tr_vals[order], width=1.0, label=f"Train (sum={int(tr_vals.sum()):,})", color="#2b6cb0")
    bottom = tr_vals[order].copy()
    if va_vals.sum() > 0:
        ax1.bar(
            x,
            va_vals[order],
            width=1.0,
            bottom=bottom,
            label=f"Val (sum={int(va_vals.sum()):,})",
            color="#63b3ed",
        )
        bottom = bottom + va_vals[order]
    if te_vals.sum() > 0:
        ax1.bar(
            x,
            te_vals[order],
            width=1.0,
            bottom=bottom,
            label=f"Test (sum={int(te_vals.sum()):,})",
            color="#ed8936",
            alpha=0.85,
        )

    imbalance_ratio = float(totals.max() / max(totals.min(), 1))
    ax1.axhline(
        float(totals.mean()),
        color="#c53030",
        linestyle="--",
        linewidth=1.3,
        label=f"Mean/Class = {totals.mean():.1f} (Min={totals.min()}, Max={totals.max()}, Ratio={imbalance_ratio:.2f}x)",
    )
    ax1.set_title("Per-Species Sample Distribution Across Splits (Sorted by Total Samples)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Sorted Species Rank")
    ax1.set_ylabel("Number of Images")
    ax1.set_xlim(-1, len(all_classes))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Bottom-8 vs Top-8 classes by train+val sample count
    dev_totals = tr_vals + va_vals
    dev_order = np.argsort(dev_totals)
    k = min(8, len(all_classes) // 2) if len(all_classes) >= 4 else len(all_classes)
    if len(all_classes) > 2 * k and k > 0:
        selected_indices = np.concatenate([dev_order[:k], dev_order[-k:]])
    else:
        selected_indices = dev_order

    def _name_for(c_idx: int) -> str:
        cid = all_classes[c_idx]
        if class_names and 0 <= cid < len(class_names):
            return class_names[cid][:26]
        return f"Class {cid}"

    labels = [_name_for(i) for i in selected_indices]
    values = dev_totals[selected_indices]
    colors = ["#e53e3e" if v < dev_totals.max() else "#3182ce" for v in values]

    y_pos = np.arange(len(selected_indices))
    bars = ax2.barh(y_pos, values, color=colors, edgecolor="white", height=0.72)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.set_xlabel("Official Train Split Images (Train + Val)")
    ax2.set_title("Lowest-Count vs. Highest-Count Bird Species in Official Train Split", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2, str(int(val)), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_image_statistics(
    image_stats: Dict[str, Any],
    save_path: Union[str, Path],
) -> Path:
    """
    Create a 4-panel diagnostic figure covering image dimensions, aspect ratios,
    empirical RGB channel statistics vs. ImageNet, and color-mode / quality checks.
    """
    out_path = _prepare_save_path(save_path)
    widths = np.asarray(image_stats.get("widths", [224]), dtype=np.float64)
    heights = np.asarray(image_stats.get("heights", [224]), dtype=np.float64)
    aspect_ratios = widths / np.maximum(heights, 1.0)
    rgb_mean = image_stats.get("rgb_mean", list(IMAGENET_MEAN))
    rgb_std = image_stats.get("rgb_std", list(IMAGENET_STD))
    mode_counts: Dict[str, int] = image_stats.get("color_modes", {"RGB": len(widths)})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    ax_wh, ax_ar, ax_rgb, ax_mode = axes.ravel()

    # 1. Width vs Height scatter
    ax_wh.scatter(widths, heights, s=10, alpha=0.35, color="#2b6cb0", edgecolors="none", label="Dataset Images")
    rect = plt.Rectangle((0, 0), 224, 224, fill=False, edgecolor="#e53e3e", linestyle="--", linewidth=1.6, label="Target Crop (224x224)")
    ax_wh.add_patch(rect)
    ax_wh.axvline(float(widths.mean()), color="#2f855a", linestyle=":", linewidth=1.3, label=f"Mean W={widths.mean():.0f}px")
    ax_wh.axhline(float(heights.mean()), color="#d69e2e", linestyle=":", linewidth=1.3, label=f"Mean H={heights.mean():.0f}px")
    ax_wh.set_title("Image Resolution Distribution (Width vs. Height)", fontweight="bold")
    ax_wh.set_xlabel("Width (px)")
    ax_wh.set_ylabel("Height (px)")
    ax_wh.legend(fontsize=8.5, loc="lower right")
    ax_wh.grid(alpha=0.25)

    # 2. Aspect Ratio histogram
    ax_ar.hist(aspect_ratios, bins=35, color="#38a169", alpha=0.8, edgecolor="white")
    ax_ar.axvline(1.0, color="#e53e3e", linestyle="--", linewidth=1.4, label="Square (1.00)")
    ax_ar.axvline(float(np.median(aspect_ratios)), color="#1a202c", linestyle="-.", linewidth=1.4, label=f"Median ({np.median(aspect_ratios):.2f})")
    ax_ar.set_title("Image Aspect Ratio (W / H) Distribution", fontweight="bold")
    ax_ar.set_xlabel("Aspect Ratio (Width / Height)")
    ax_ar.set_ylabel("Frequency")
    ax_ar.legend(fontsize=8.5)
    ax_ar.grid(alpha=0.25)

    # 3. Empirical RGB Mean & Std vs ImageNet
    channels = ["R", "G", "B"]
    x_pos = np.arange(3)
    bar_w = 0.2
    ax_rgb.bar(x_pos - 1.5 * bar_w, rgb_mean, width=bar_w, label="Dataset Mean", color="#3182ce")
    ax_rgb.bar(x_pos - 0.5 * bar_w, IMAGENET_MEAN, width=bar_w, label="ImageNet Mean", color="#90cdf4")
    ax_rgb.bar(x_pos + 0.5 * bar_w, rgb_std, width=bar_w, label="Dataset Std", color="#dd6b20")
    ax_rgb.bar(x_pos + 1.5 * bar_w, IMAGENET_STD, width=bar_w, label="ImageNet Std", color="#fbd38d")
    ax_rgb.set_xticks(x_pos)
    ax_rgb.set_xticklabels(channels, fontweight="bold")
    ax_rgb.set_ylim(0, 0.75)
    ax_rgb.set_ylabel("Normalized Pixel Value [0, 1]")
    ax_rgb.set_title("Empirical RGB Channel Stats vs. ImageNet Normalization", fontweight="bold")
    ax_rgb.legend(fontsize=8.5, ncol=2)
    ax_rgb.grid(axis="y", alpha=0.25)

    # 4. Color modes & data integrity anomalies
    anomaly_labels = [f"Mode: {m}" for m in mode_counts.keys()] + [
        "Corrupted",
        "Exact Dups",
        "Cross-Split Dups",
    ]
    anomaly_vals = list(mode_counts.values()) + [
        int(image_stats.get("corrupted_count", 0)),
        int(image_stats.get("exact_duplicate_groups", 0)),
        int(image_stats.get("cross_split_duplicate_groups", 0)),
    ]
    bar_colors = ["#2b6cb0" if lbl == "Mode: RGB" else "#e53e3e" for lbl in anomaly_labels]
    bars = ax_mode.bar(np.arange(len(anomaly_labels)), anomaly_vals, color=bar_colors, alpha=0.88)
    ax_mode.set_xticks(np.arange(len(anomaly_labels)))
    ax_mode.set_xticklabels(anomaly_labels, rotation=22, ha="right", fontsize=8.5)
    ax_mode.set_ylabel("Count")
    ax_mode.set_title("Color Mode & Data Quality / Duplicate Audit", fontweight="bold")
    ax_mode.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, anomaly_vals):
        ax_mode.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max(anomaly_vals, default=1) * 0.015, 0.2),
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.suptitle("CUB-200-2011 Visual Data Quality & Image Statistics", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_caption_statistics(
    caption_stats: Dict[str, Any],
    save_path: Union[str, Path],
) -> Path:
    """
    Create a 4-panel figure summarizing caption word-length distributions, top
    descriptive bird vocabulary, vocabulary/noise cleaning impact, and captions per image.
    """
    out_path = _prepare_save_path(save_path)
    wl = caption_stats.get("word_length_stats", {})
    vs = caption_stats.get("vocabulary_stats", {})
    ca = caption_stats.get("cleaning_audit", {})
    cpi = caption_stats.get("captions_per_image", {})

    raw_lengths = np.asarray(wl.get("raw_lengths", [10]), dtype=np.int32)
    clean_lengths = np.asarray(wl.get("cleaned_lengths", [10]), dtype=np.int32)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax_len, ax_vocab, ax_clean, ax_cpi = axes.ravel()

    # 1. Word-length histogram
    max_bin = int(min(max(clean_lengths.max(initial=20), 15), 45))
    bins = np.arange(1, max_bin + 2) - 0.5
    ax_len.hist(raw_lengths, bins=bins, alpha=0.45, label="Raw Captions", color="#a0aec0", edgecolor="white")
    ax_len.hist(clean_lengths, bins=bins, alpha=0.75, label="Cleaned Captions", color="#2b6cb0", edgecolor="white")
    p95 = float(wl.get("cleaned_p95", np.percentile(clean_lengths, 95)))
    mean_l = float(wl.get("cleaned_mean", clean_lengths.mean()))
    ax_len.axvline(mean_l, color="#2f855a", linestyle="--", linewidth=1.5, label=f"Mean = {mean_l:.1f} words")
    ax_len.axvline(p95, color="#e53e3e", linestyle="-.", linewidth=1.5, label=f"95th Pct = {p95:.0f} words")
    ax_len.set_title("Caption Length Distribution (Words per Caption)", fontweight="bold")
    ax_len.set_xlabel("Words per Caption")
    ax_len.set_ylabel("Number of Captions")
    ax_len.legend(fontsize=8.5)
    ax_len.grid(alpha=0.25)

    # 2. Top-20 descriptive words (excluding stopwords)
    top_desc: Dict[str, int] = vs.get("top_30_descriptive_words", {})
    items = list(top_desc.items())[:20][::-1]
    if items:
        words, freqs = zip(*items)
        ax_vocab.barh(np.arange(len(words)), freqs, color="#319795", height=0.75)
        ax_vocab.set_yticks(np.arange(len(words)))
        ax_vocab.set_yticklabels(words, fontsize=8.5)
    ax_vocab.set_title("Top-20 Descriptive Bird Words (Stopwords Removed)", fontweight="bold")
    ax_vocab.set_xlabel("Corpus Frequency")
    ax_vocab.grid(axis="x", alpha=0.25)

    # 3. Vocabulary & Cleaning Audit Summary
    raw_v = int(vs.get("raw_vocab_size", 0))
    clean_v = int(vs.get("cleaned_vocab_size", 0))
    red_pct = float(vs.get("vocab_reduction_pct", 0.0))
    audit_names = [
        "Raw Vocab",
        "Clean Vocab",
        "Typo Fixes",
        "Empty Lines",
        "Intra-Img Dups",
        "Too Short (<2w)",
    ]
    audit_vals = [
        raw_v,
        clean_v,
        int(ca.get("total_typo_fixes_applied", 0)),
        int(ca.get("empty_lines_removed", 0)),
        int(ca.get("dropped_intra_image_duplicates", 0)),
        int(ca.get("dropped_too_short", 0)),
    ]
    bar_cols = ["#718096", "#2b6cb0", "#dd6b20", "#e53e3e", "#d69e2e", "#805ad5"]
    bars = ax_clean.bar(np.arange(len(audit_names)), audit_vals, color=bar_cols, alpha=0.88)
    ax_clean.set_xticks(np.arange(len(audit_names)))
    ax_clean.set_xticklabels(audit_names, rotation=22, ha="right", fontsize=8.5)
    ax_clean.set_ylabel("Count")
    ax_clean.set_title(f"Caption Preprocessing Impact (Vocab Reduced by {red_pct:.1f}%)", fontweight="bold")
    ax_clean.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, audit_vals):
        ax_clean.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max(audit_vals, default=1) * 0.015, 0.2),
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # 4. Captions-per-image distribution
    raw_dist: Dict[str, int] = cpi.get("raw_distribution", {"10": caption_stats.get("num_records", 0)})
    dist_keys = sorted(raw_dist.keys(), key=lambda k: int(k))
    dist_vals = [raw_dist[k] for k in dist_keys]
    cols_cpi = ["#2f855a" if k == "10" else "#e53e3e" for k in dist_keys]
    bars_cpi = ax_cpi.bar(dist_keys, dist_vals, color=cols_cpi, width=0.55)
    ax_cpi.set_title("Captions per Image Record (Expected = 10)", fontweight="bold")
    ax_cpi.set_xlabel("Captions Found After Splitting on '\\n'")
    ax_cpi.set_ylabel("Number of Images")
    ax_cpi.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars_cpi, dist_vals):
        ax_cpi.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(max(dist_vals, default=1) * 0.015, 0.2),
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.suptitle("CUB-200-2011 Natural Language Description EDA & Cleaning Audit", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_augmentation_examples(
    pil_image: Image.Image,
    save_path: Union[str, Path],
    train_transform: Optional[Callable] = None,
    image_size: int = 224,
    augmentation: str = "fine_grained",
    num_views: int = 5,
    class_name: Optional[str] = None,
) -> Path:
    """
    Visualize an original bird image alongside `num_views` stochastic augmented
    samples produced by the training augmentation pipeline.
    """
    out_path = _prepare_save_path(save_path)
    if train_transform is None:
        train_transform, _ = get_cub200_transforms(
            image_size=image_size,
            augmentation=augmentation,
        )

    rgb_orig = pil_image.convert("RGB")
    total_cols = 1 + max(1, num_views)
    fig, axes = plt.subplots(1, total_cols, figsize=(3.1 * total_cols, 3.6))
    axes_flat = np.atleast_1d(axes).ravel()

    axes_flat[0].imshow(np.asarray(rgb_orig))
    axes_flat[0].set_title(f"Original\n({rgb_orig.size[0]}x{rgb_orig.size[1]})", fontsize=9.5, fontweight="bold")
    axes_flat[0].axis("off")

    for i in range(1, total_cols):
        aug_tensor = train_transform(rgb_orig)
        aug_rgb = denormalize_tensor(aug_tensor)
        axes_flat[i].imshow(aug_rgb)
        axes_flat[i].set_title(f"Aug View #{i}\n({augmentation})", fontsize=9)
        axes_flat[i].axis("off")

    species_suffix = f" — {class_name}" if class_name else ""
    fig.suptitle(
        f"Data Augmentation Strategy Preview ({augmentation}){species_suffix}",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path

