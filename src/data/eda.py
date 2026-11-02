"""
End-to-end Task 1.1 Data Understanding, Exploratory Data Analysis (EDA),
Anomaly Detection, Caption Cleaning, and Matplotlib Visualization Pipeline.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from PIL import Image

from .captions import analyze_captions
from .cub200 import (
    CUB200Dataset,
    HFCUB200Dataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_cub200_transforms,
    stratified_split_samples,
)

try:
    from src.visualization.eda_plots import (
        plot_augmentation_examples,
        plot_caption_statistics,
        plot_class_distribution,
        plot_image_statistics,
        plot_sample_images,
    )
except ImportError:
    from visualization.eda_plots import (
        plot_augmentation_examples,
        plot_caption_statistics,
        plot_class_distribution,
        plot_image_statistics,
        plot_sample_images,
    )


def _open_pil_detached(source_img: Any) -> Image.Image:
    """Open an image path safely without leaking file handles on Windows, or return in-memory PIL image."""
    if isinstance(source_img, (str, Path)):
        with Image.open(source_img) as opened:
            opened.load()
            return opened.copy()
    if isinstance(source_img, Image.Image):
        source_img.load()
        return source_img
    return Image.fromarray(np.asarray(source_img))


def _compute_image_md5(img: Image.Image) -> str:
    """Compute deterministic MD5 fingerprint of decoded image mode, size, and pixel bytes."""
    hasher = hashlib.md5()
    hasher.update(f"{img.mode}:{img.size[0]}x{img.size[1]}:".encode("utf-8"))
    hasher.update(img.tobytes())
    return hasher.hexdigest()


def _audit_images(
    records: Sequence[Dict[str, Any]],
    max_scan: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scan image records for dimensions, color modes, RGB pixel statistics,
    corrupted files, low-contrast/dark outliers, and exact duplicates.

    Each item in `records` expects:
      - `'split'`: `'train'` or `'test'`
      - `'id'`: identifier / file name string
      - `'loader'`: zero-arg callable returning a raw `PIL.Image.Image` (before `.convert('RGB')`)
    """
    limit = len(records) if max_scan is None else min(len(records), int(max_scan))
    widths: List[int] = []
    heights: List[int] = []
    color_modes: Counter = Counter()
    corrupted_files: List[Dict[str, str]] = []
    non_rgb_files: List[Dict[str, str]] = []
    low_res_files: List[Dict[str, Any]] = []
    extreme_exposure_files: List[Dict[str, Any]] = []
    md5_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for idx in range(limit):
        rec = records[idx]
        rec_id = str(rec["id"])
        split_name = str(rec["split"])
        try:
            raw_img = rec["loader"]()
            raw_img.load()
        except Exception as exc:
            corrupted_files.append({"split": split_name, "id": rec_id, "error": str(exc)})
            continue

        mode = str(raw_img.mode)
        w, h = raw_img.size
        widths.append(int(w))
        heights.append(int(h))
        color_modes[mode] += 1

        if mode != "RGB":
            non_rgb_files.append({"split": split_name, "id": rec_id, "mode": mode})

        if min(w, h) < 150 and len(low_res_files) < 25:
            low_res_files.append({"split": split_name, "id": rec_id, "width": w, "height": h})

        digest = _compute_image_md5(raw_img)
        md5_map[digest].append({"split": split_name, "id": rec_id})

        rgb_img = raw_img.convert("RGB")
        # Compute RGB stats on training split only to avoid test-set statistics leakage
        arr = np.asarray(rgb_img, dtype=np.float64) / 255.0
        brightness = float(arr.mean())
        contrast = float(arr.std())
        if (brightness < 0.08 or brightness > 0.92 or contrast < 0.05) and len(extreme_exposure_files) < 25:
            extreme_exposure_files.append(
                {
                    "split": split_name,
                    "id": rec_id,
                    "brightness": round(brightness, 4),
                    "contrast": round(contrast, 4),
                }
            )

        if split_name == "train":
            flat = arr.reshape(-1, 3)
            channel_sum += flat.sum(axis=0)
            channel_sq_sum += np.square(flat).sum(axis=0)
            total_pixels += flat.shape[0]

    if total_pixels > 0:
        rgb_mean = (channel_sum / total_pixels).tolist()
        rgb_var = np.maximum(channel_sq_sum / total_pixels - np.square(rgb_mean), 0.0)
        rgb_std = np.sqrt(rgb_var).tolist()
    else:
        rgb_mean = list(IMAGENET_MEAN)
        rgb_std = list(IMAGENET_STD)

    duplicate_groups = [items for items in md5_map.values() if len(items) > 1]
    cross_split_groups = [
        items for items in duplicate_groups if len({it["split"] for it in items}) > 1
    ]

    w_arr = np.asarray(widths if widths else [0], dtype=np.float64)
    h_arr = np.asarray(heights if heights else [0], dtype=np.float64)
    ar_arr = w_arr / np.maximum(h_arr, 1.0)

    return {
        "scanned_images": limit,
        "widths": widths,
        "heights": heights,
        "width_summary": {
            "min": int(w_arr.min()),
            "max": int(w_arr.max()),
            "mean": round(float(w_arr.mean()), 2),
            "median": round(float(np.median(w_arr)), 2),
            "std": round(float(w_arr.std()), 2),
        },
        "height_summary": {
            "min": int(h_arr.min()),
            "max": int(h_arr.max()),
            "mean": round(float(h_arr.mean()), 2),
            "median": round(float(np.median(h_arr)), 2),
            "std": round(float(h_arr.std()), 2),
        },
        "aspect_ratio_summary": {
            "min": round(float(ar_arr.min()), 3),
            "max": round(float(ar_arr.max()), 3),
            "mean": round(float(ar_arr.mean()), 3),
            "median": round(float(np.median(ar_arr)), 3),
        },
        "rgb_mean": [round(float(x), 4) for x in rgb_mean],
        "rgb_std": [round(float(x), 4) for x in rgb_std],
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "color_modes": {str(k): int(v) for k, v in color_modes.items()},
        "non_rgb_count": len(non_rgb_files),
        "non_rgb_samples": non_rgb_files[:25],
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "low_resolution_samples": low_res_files,
        "extreme_exposure_samples": extreme_exposure_files,
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "duplicate_examples": duplicate_groups[:15],
        "cross_split_duplicate_examples": cross_split_groups[:15],
    }


def _summarize_class_distribution(
    train_labels: Sequence[int],
    val_labels: Sequence[int],
    test_labels: Sequence[int],
    class_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[int, int], Dict[int, int], Dict[int, int]]:
    tr_c = Counter(int(x) for x in train_labels)
    va_c = Counter(int(x) for x in val_labels)
    te_c = Counter(int(x) for x in test_labels)

    all_ids = sorted(set(tr_c.keys()) | set(va_c.keys()) | set(te_c.keys()))
    if not all_ids:
        all_ids = list(range(len(class_names)))

    official_train_arr = np.array([tr_c.get(i, 0) + va_c.get(i, 0) for i in all_ids], dtype=np.int32)
    test_arr = np.array([te_c.get(i, 0) for i in all_ids], dtype=np.int32)
    total_arr = official_train_arr + test_arr

    max_train = int(official_train_arr.max()) if len(official_train_arr) else 0
    min_train = int(official_train_arr.min()) if len(official_train_arr) else 0
    underrep = [
        {
            "class_id": int(cid),
            "species": class_names[cid] if cid < len(class_names) else f"Class_{cid}",
            "official_train_count": int(tr_c.get(cid, 0) + va_c.get(cid, 0)),
            "test_count": int(te_c.get(cid, 0)),
        }
        for cid in all_ids
        if (tr_c.get(cid, 0) + va_c.get(cid, 0)) < max_train
    ]

    summary = {
        "num_classes": len(all_ids),
        "split_sizes": {
            "train_subsplit": len(train_labels),
            "val_subsplit": len(val_labels),
            "official_train_total": len(train_labels) + len(val_labels),
            "official_test_total": len(test_labels),
            "grand_total": len(train_labels) + len(val_labels) + len(test_labels),
        },
        "official_train_per_class": {
            "min": min_train,
            "max": max_train,
            "mean": round(float(official_train_arr.mean()), 2) if len(official_train_arr) else 0.0,
            "std": round(float(official_train_arr.std()), 3) if len(official_train_arr) else 0.0,
            "imbalance_ratio": round(float(max_train / max(min_train, 1)), 3),
        },
        "total_per_class": {
            "min": int(total_arr.min()) if len(total_arr) else 0,
            "max": int(total_arr.max()) if len(total_arr) else 0,
            "mean": round(float(total_arr.mean()), 2) if len(total_arr) else 0.0,
            "std": round(float(total_arr.std()), 3) if len(total_arr) else 0.0,
            "imbalance_ratio": round(
                float(total_arr.max() / max(int(total_arr.min()), 1)), 3
            )
            if len(total_arr)
            else 1.0,
        },
        "underrepresented_train_classes_count": len(underrep),
        "underrepresented_train_classes": underrep[:20],
    }
    return summary, dict(tr_c), dict(va_c), dict(te_c)


def _write_markdown_eda_report(report: Dict[str, Any], md_path: Path) -> Path:
    """Render a comprehensive, report-ready Markdown summary of Task 1.1 findings."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    cd = report["class_distribution"]
    im = report["image_statistics"]
    cap = report.get("caption_statistics")
    figs = report.get("generated_figures", {})

    lines = [
        "# Task 1.1: CUB-200-2011 Exploratory Data Analysis & Preprocessing Report",
        "",
        f"- **Dataset Source**: `{report['dataset_source']}`",
        f"- **Total Classes**: `{cd['num_classes']}` bird species",
        f"- **Split Sizes**: Train = `{cd['split_sizes']['train_subsplit']:,}` | "
        f"Val = `{cd['split_sizes']['val_subsplit']:,}` (stratified from `{cd['split_sizes']['official_train_total']:,}`) | "
        f"Test = `{cd['split_sizes']['official_test_total']:,}` | "
        f"Total = `{cd['split_sizes']['grand_total']:,}`",
        "",
        "## 1. Class Distribution & Balance",
        f"- Official training images per class: min = `{cd['official_train_per_class']['min']}`, "
        f"max = `{cd['official_train_per_class']['max']}`, "
        f"mean = `{cd['official_train_per_class']['mean']}`, "
        f"imbalance ratio = `{cd['official_train_per_class']['imbalance_ratio']}x`.",
        f"- Total (train+test) images per class: min = `{cd['total_per_class']['min']}`, "
        f"max = `{cd['total_per_class']['max']}`, "
        f"imbalance ratio = `{cd['total_per_class']['imbalance_ratio']}x`.",
        f"- Classes below maximum training count: `{cd['underrepresented_train_classes_count']}` species.",
        "- **Modeling Decision**: Because class counts are nearly uniform (~29–30 training samples/species), "
        "standard unweighted Cross-Entropy with **per-class stratified train/val splitting** (`val_split=0.2`) "
        "preserves exact species representation without requiring oversampling.",
        "",
        "## 2. Image Dimensions, Channels & Data Integrity Audit",
        f"- **Scanned Images**: `{im['scanned_images']:,}`",
        f"- **Width (px)**: min = `{im['width_summary']['min']}`, max = `{im['width_summary']['max']}`, "
        f"mean = `{im['width_summary']['mean']}`, median = `{im['width_summary']['median']}`",
        f"- **Height (px)**: min = `{im['height_summary']['min']}`, max = `{im['height_summary']['max']}`, "
        f"mean = `{im['height_summary']['mean']}`, median = `{im['height_summary']['median']}`",
        f"- **Aspect Ratio (W/H)**: median = `{im['aspect_ratio_summary']['median']}` "
        f"(range `{im['aspect_ratio_summary']['min']}` – `{im['aspect_ratio_summary']['max']}`)",
        f"- **Empirical Train RGB Mean / Std**: mean = `{im['rgb_mean']}`, std = `{im['rgb_std']}` "
        f"(vs. ImageNet mean `{im['imagenet_mean']}`, std `{im['imagenet_std']}`)",
        f"- **Color Mode Anomalies**: `{im['color_modes']}` (`{im['non_rgb_count']}` non-RGB grayscale/other images converted via `.convert('RGB')`)",
        f"- **Corrupted Files**: `{im['corrupted_count']}`",
        f"- **Exact Duplicate Groups**: `{im['exact_duplicate_groups']}` total (`{im['cross_split_duplicate_groups']}` cross-split train/test overlap groups)",
        "",
    ]

    if cap is not None:
        vs = cap["vocabulary_stats"]
        wl = cap["word_length_stats"]
        ca = cap["cleaning_audit"]
        cpi = cap["captions_per_image"]
        lines.extend(
            [
                "## 3. Natural Language Captions (`description` Column) & Preprocessing",
                f"- **Records Analyzed**: `{cap['num_records']:,}` (`{ca['total_raw_captions']:,}` raw captions -> `{ca['total_cleaned_captions']:,}` cleaned captions)",
                f"- **Captions per Image**: `{cpi['exact_10_raw_records']:,}` records had exactly 10 captions; `{cpi['non_10_raw_records_count']}` records differed.",
                f"- **Caption Word Length (Cleaned)**: mean = `{wl['cleaned_mean']:.2f}` words, median = `{wl['cleaned_median']:.1f}`, "
                f"95th percentile = `{wl['cleaned_p95']:.0f}` words (range `{wl['cleaned_min']}` – `{wl['cleaned_max']}`).",
                f"- **Vocabulary Compression**: Raw tokens = `{vs['raw_vocab_size']:,}` -> Cleaned vocabulary = `{vs['cleaned_vocab_size']:,}` "
                f"(`{vs['vocab_reduction_pct']:.1f}%` noise reduction).",
                f"- **Cleaning Actions Applied**: `{ca['total_typo_fixes_applied']:,}` domain typo/spelling fixes, "
                f"`{ca['empty_lines_removed']:,}` empty lines stripped, "
                f"`{ca['dropped_intra_image_duplicates']:,}` intra-image duplicate captions removed, "
                f"`{ca['dropped_too_short']:,}` degenerate `<2`-word fragments dropped.",
                "",
                "### Caption Preprocessing Pipeline Justification",
                "1. **Newline Splitting (`\\n`)**: Separates the raw `description` block into 10 independent reference sentences per image and drops empty lines.",
                "2. **Unicode NFKC & Lowercasing**: Unifies case variants (`'Bird'` vs `'bird'`) to reduce vocabulary sparsity.",
                "3. **Hyphen Separation & Punctuation Stripping**: Splits compound modifiers (`'black-billed'` -> `'black billed'`) and strips trailing periods/commas so `'wings.'` and `'wings'` map to one token.",
                "4. **Crowdsourced Typo & UK/US Normalization**: Corrects recurring AMT bird-anatomy misspellings (`'feathres'` -> `'feathers'`, `'beack'` -> `'beak'`, `'brest'` -> `'breast'`, `'grey'` -> `'gray'`, `'colour'` -> `'color'`).",
                "5. **Intra-Image Deduplication & Minimum-Length Filtering**: Removes accidental repeated lines within the same bird's 10 descriptions and `<2`-word noise fragments.",
                "",
            ]
        )

    lines.extend(
        [
            "## 4. Data Augmentation Strategy Justification",
            "- **`fine_grained` Augmentation Pipeline**:",
            "  - `RandomResizedCrop(224, scale=(0.70, 1.0), ratio=(0.85, 1.15))`: Handles variable telephoto zoom levels while keeping diagnostic anatomical parts (bill, crown, wing bars) inside the frame.",
            "  - `RandomHorizontalFlip(p=0.5)` & `RandomRotation(15 deg)`: Models bilateral bird symmetry and natural branch/perch tilt.",
            "  - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)`: Models outdoor daylight/shadow variations while keeping **`hue=0.02`** so species-defining plumage colors (e.g., *Scarlet Tanager* vs. *Summer Tanager*, *Yellow Warbler* vs. *Blue-winged Warbler*) are never corrupted.",
            "",
            "## 5. Generated Figures",
        ]
    )
    for fig_key, fig_path in figs.items():
        lines.append(f"- **{fig_key}**: `{fig_path}`")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def run_eda(
    cfg: Optional[Union[Dict[str, Any], DictConfig]] = None,
    hf_train_data: Optional[Any] = None,
    hf_test_data: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute full Task 1.1 EDA, caption preprocessing, anomaly detection, and
    Matplotlib figure generation.

    Supports:
      - `eda.source = 'huggingface'`: Loads `Multimodal-Fatima/CUB_train` & `CUB_test`
      - `eda.source = 'local'`: Loads local Caltech `CUB_200_2011`
      - `eda.source = 'auto'`: Tries local or Hugging Face automatically (prefers HF if captions requested)
      - Direct in-memory `hf_train_data` / `hf_test_data` injection for fast unit testing.
    """
    cfg = cfg or {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, (dict, DictConfig)) else {}
    eda_cfg = cfg.get("eda", {}) if isinstance(cfg, (dict, DictConfig)) else {}

    data_dir = Path(data_cfg.get("data_dir", "data"))
    image_size = int(data_cfg.get("image_size", 224))
    val_split = float(data_cfg.get("val_split", 0.2))
    seed = int(data_cfg.get("seed", cfg.get("seed", 42)))

    source = str(eda_cfg.get("source", data_cfg.get("source", "auto"))).strip().lower()
    output_dir = Path(eda_cfg.get("output_dir", "outputs/eda"))
    figures_dir = Path(eda_cfg.get("figures_dir", "outputs/figures/eda"))
    max_image_scan = eda_cfg.get("max_image_scan", None)
    save_cleaned_captions = bool(eda_cfg.get("save_cleaned_captions", True))
    num_sample_images = int(eda_cfg.get("num_sample_images", 8))
    num_aug_views = int(eda_cfg.get("num_augmentation_views", 5))

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    use_hf = False
    if hf_train_data is not None:
        use_hf = True
    elif source in ("hf", "huggingface"):
        use_hf = True
    elif source == "auto":
        # Prefer Hugging Face so Task 1.1 caption EDA runs automatically; fall back to local if offline
        try:
            import datasets  # noqa: F401
            use_hf = True
        except ImportError:
            use_hf = False

    train_descriptions: List[str] = []
    test_descriptions: List[str] = []
    image_audit_records: List[Dict[str, Any]] = []
    sample_cards: List[Dict[str, Any]] = []
    active_source = "huggingface" if use_hf else "local"

    if use_hf:
        try:
            hf_train_ds = HFCUB200Dataset(
                train=True,
                transform=None,
                cache_dir=data_dir,
                hf_dataset=hf_train_data,
            )
            hf_test_ds = HFCUB200Dataset(
                train=False,
                transform=None,
                cache_dir=data_dir,
                hf_dataset=hf_test_data if hf_test_data is not None else (hf_train_data if hf_train_data is not None else None),
                label_to_idx=hf_train_ds.label_to_idx,
                class_names=hf_train_ds.class_names,
            )
            class_names = hf_train_ds.class_names
        except Exception as exc:
            if source != "auto":
                raise
            print(f"[EDA] Hugging Face dataset unavailable ({exc}); falling back to local CUB-200-2011.")
            use_hf = False
            active_source = "local"

    if use_hf:
        train_pairs, val_pairs = stratified_split_samples(
            hf_train_ds.samples, val_ratio=val_split, seed=seed
        )
        train_labels = [lbl for _, lbl in train_pairs]
        val_labels = [lbl for _, lbl in val_pairs]
        test_labels = [lbl for _, lbl in hf_test_ds.samples]

        for idx, _ in hf_train_ds.samples:
            rec = hf_train_ds.hf_dataset[idx]
            train_descriptions.append(str(rec.get("description", "")))
            rec_id = str(rec.get("file_name", rec.get("id", f"train_{idx}")))
            image_audit_records.append(
                {
                    "split": "train",
                    "id": rec_id,
                    "loader": (lambda r=rec: _open_pil_detached(r["image"])),
                }
            )

        if hf_test_data is not None or hf_train_data is None:
            for idx, _ in hf_test_ds.samples:
                rec = hf_test_ds.hf_dataset[idx]
                test_descriptions.append(str(rec.get("description", "")))
                rec_id = str(rec.get("file_name", rec.get("id", f"test_{idx}")))
                image_audit_records.append(
                    {
                        "split": "test",
                        "id": rec_id,
                        "loader": (lambda r=rec: _open_pil_detached(r["image"])),
                    }
                )

        step = max(1, len(hf_train_ds) // max(1, num_sample_images))
        for sample_i in range(0, len(hf_train_ds), step):
            if len(sample_cards) >= num_sample_images:
                break
            pil_img, lbl = hf_train_ds[sample_i]
            cleaned_caps = hf_train_ds.get_cleaned_captions(sample_i)
            sample_cards.append(
                {
                    "image": pil_img,
                    "class_name": class_names[lbl] if lbl < len(class_names) else f"Class_{lbl}",
                    "caption": cleaned_caps[0] if cleaned_caps else None,
                }
            )
    else:
        download = bool(data_cfg.get("download", True))
        local_train_ds = CUB200Dataset(root=data_dir, train=True, transform=None, download=download)
        local_test_ds = CUB200Dataset(root=data_dir, train=False, transform=None, download=False)
        class_names = local_train_ds.class_names

        train_pairs, val_pairs = stratified_split_samples(
            local_train_ds.samples, val_ratio=val_split, seed=seed
        )
        train_labels = [lbl for _, lbl in train_pairs]
        val_labels = [lbl for _, lbl in val_pairs]
        test_labels = [lbl for _, lbl in local_test_ds.samples]

        for img_path, _ in local_train_ds.samples:
            image_audit_records.append(
                {
                    "split": "train",
                    "id": str(img_path.name),
                    "loader": (lambda p=img_path: _open_pil_detached(p)),
                }
            )
        for img_path, _ in local_test_ds.samples:
            image_audit_records.append(
                {
                    "split": "test",
                    "id": str(img_path.name),
                    "loader": (lambda p=img_path: _open_pil_detached(p)),
                }
            )

        step = max(1, len(local_train_ds) // max(1, num_sample_images))
        for sample_i in range(0, len(local_train_ds), step):
            if len(sample_cards) >= num_sample_images:
                break
            pil_img, lbl = local_train_ds[sample_i]
            sample_cards.append(
                {
                    "image": pil_img,
                    "class_name": class_names[lbl] if lbl < len(class_names) else f"Class_{lbl}",
                    "caption": None,
                }
            )

    class_dist_summary, tr_counts, va_counts, te_counts = _summarize_class_distribution(
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        class_names=class_names,
    )
    image_stats = _audit_images(image_audit_records, max_scan=max_image_scan)

    generated_figures: Dict[str, str] = {}
    generated_figures["sample_images"] = str(
        plot_sample_images(sample_cards, save_path=figures_dir / "sample_birds_and_captions.png")
    )
    generated_figures["class_distribution"] = str(
        plot_class_distribution(
            train_counts=tr_counts,
            val_counts=va_counts,
            test_counts=te_counts,
            class_names=class_names,
            save_path=figures_dir / "class_distribution.png",
        )
    )
    generated_figures["image_statistics"] = str(
        plot_image_statistics(
            image_stats=image_stats,
            save_path=figures_dir / "image_statistics_and_quality.png",
        )
    )

    if sample_cards:
        hero_img = sample_cards[0]["image"]
        hero_cls = sample_cards[0]["class_name"]
        fg_tf, _ = get_cub200_transforms(image_size=image_size, augmentation="fine_grained")
        basic_tf, _ = get_cub200_transforms(image_size=image_size, augmentation="basic")
        generated_figures["augmentation_fine_grained"] = str(
            plot_augmentation_examples(
                pil_image=hero_img,
                train_transform=fg_tf,
                image_size=image_size,
                augmentation="fine_grained",
                num_views=num_aug_views,
                class_name=hero_cls,
                save_path=figures_dir / "augmentation_examples_fine_grained.png",
            )
        )
        generated_figures["augmentation_basic"] = str(
            plot_augmentation_examples(
                pil_image=hero_img,
                train_transform=basic_tf,
                image_size=image_size,
                augmentation="basic",
                num_views=num_aug_views,
                class_name=hero_cls,
                save_path=figures_dir / "augmentation_examples_basic.png",
            )
        )

    caption_report: Optional[Dict[str, Any]] = None
    all_descriptions = train_descriptions + test_descriptions
    if all_descriptions and any(d.strip() for d in all_descriptions):
        caption_report, _ = analyze_captions(all_descriptions)
        generated_figures["caption_statistics"] = str(
            plot_caption_statistics(
                caption_stats=caption_report,
                save_path=figures_dir / "caption_statistics_and_cleaning.png",
            )
        )
        if save_cleaned_captions:
            _, cleaned_train = analyze_captions(train_descriptions)
            (output_dir / "cleaned_captions_train.json").write_text(
                json.dumps(cleaned_train, indent=2), encoding="utf-8"
            )
            if test_descriptions:
                _, cleaned_test = analyze_captions(test_descriptions)
                (output_dir / "cleaned_captions_test.json").write_text(
                    json.dumps(cleaned_test, indent=2), encoding="utf-8"
                )

    # Compact JSON copy without huge raw per-image lists
    compact_image_stats = {
        k: v for k, v in image_stats.items() if k not in ("widths", "heights")
    }
    compact_caption_stats: Optional[Dict[str, Any]] = None
    if caption_report is not None:
        compact_wl = {
            k: v
            for k, v in caption_report["word_length_stats"].items()
            if k not in ("raw_lengths", "cleaned_lengths")
        }
        compact_caption_stats = {**caption_report, "word_length_stats": compact_wl}

    full_report: Dict[str, Any] = {
        "dataset_source": active_source,
        "class_distribution": class_dist_summary,
        "image_statistics": compact_image_stats,
        "caption_statistics": compact_caption_stats,
        "generated_figures": generated_figures,
    }

    json_path = output_dir / "eda_report.json"
    json_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    md_path = _write_markdown_eda_report(full_report, output_dir / "eda_summary.md")
    full_report["report_json_path"] = str(json_path)
    full_report["report_markdown_path"] = str(md_path)

    print(f"[EDA] Completed Task 1.1 analysis ({active_source}).")
    print(f"[EDA] Summary Markdown : {md_path}")
    print(f"[EDA] JSON Report      : {json_path}")
    print(f"[EDA] Figures Directory: {figures_dir}")
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task 1.1 CUB-200-2011 EDA and Caption Cleaning.")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "huggingface", "local"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/eda")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures/eda")
    parser.add_argument("--max-scan", type=int, default=None)
    args = parser.parse_args()

    run_eda(
        cfg={
            "data": {"data_dir": args.data_dir, "source": args.source},
            "eda": {
                "source": args.source,
                "output_dir": args.output_dir,
                "figures_dir": args.figures_dir,
                "max_image_scan": args.max_scan,
            },
        }
    )
