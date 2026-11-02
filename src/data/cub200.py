import argparse
from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

from .captions import clean_caption, split_raw_description

CUB_ARCHIVE_FILENAME = "CUB_200_2011.tgz"
CUB_EXTRACTED_FOLDER = "CUB_200_2011"
CUB_MD5 = "97eceeb196236b17998738112f37df78"
CUB_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz",
]

HF_CUB_TRAIN_REPO = "Multimodal-Fatima/CUB_train"
HF_CUB_TEST_REPO = "Multimodal-Fatima/CUB_test"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_cub200_extracted(cub_dir: Path) -> bool:
    required = [
        cub_dir / "images.txt",
        cub_dir / "image_class_labels.txt",
        cub_dir / "train_test_split.txt",
        cub_dir / "images",
    ]
    return all(p.exists() for p in required)


def _format_species_name(raw_name: str) -> str:
    """Clean CUB class names like '001.Black_footed_Albatross' -> 'Black footed Albatross'."""
    name = raw_name.strip()
    if "." in name and name.split(".", 1)[0].isdigit():
        name = name.split(".", 1)[1]
    return name.replace("_", " ").strip()


def load_cub200_class_names(cub_dir: Path, num_classes: int = 200) -> List[str]:
    """
    Load ordered 0-indexed bird species names from `classes.txt` or infer them
    from folder prefixes in `images.txt`.
    """
    classes_file = cub_dir / "classes.txt"
    names_by_idx: Dict[int, str] = {}

    if classes_file.is_file():
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    names_by_idx[int(parts[0]) - 1] = _format_species_name(parts[1])

    if not names_by_idx:
        images_file = cub_dir / "images.txt"
        labels_file = cub_dir / "image_class_labels.txt"
        if images_file.is_file() and labels_file.is_file():
            img_folders: Dict[int, str] = {}
            with open(images_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        folder = parts[1].split("/")[0]
                        img_folders[int(parts[0])] = _format_species_name(folder)
            with open(labels_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        img_id, cls_id = int(parts[0]), int(parts[1]) - 1
                        if cls_id not in names_by_idx and img_id in img_folders:
                            names_by_idx[cls_id] = img_folders[img_id]

    total = max(num_classes, (max(names_by_idx.keys()) + 1) if names_by_idx else num_classes)
    return [names_by_idx.get(i, f"Species_{i:03d}") for i in range(total)]


def download_cub200(
    root: Union[str, Path] = "data",
    remove_finished: bool = False,
) -> Path:
    """
    Download and extract the CUB-200-2011 dataset into `root/CUB_200_2011`.
    Skips downloading if the extracted dataset files already exist.
    """
    root_path = Path(root).expanduser().resolve()
    cub_dir = root_path / CUB_EXTRACTED_FOLDER

    if _is_cub200_extracted(cub_dir):
        print(f"[CUB-200] Dataset already exists at: {cub_dir}")
        return cub_dir

    root_path.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None

    for url in CUB_URLS:
        try:
            print(f"[CUB-200] Downloading dataset from {url} into {root_path} ...")
            download_and_extract_archive(
                url=url,
                download_root=str(root_path),
                extract_root=str(root_path),
                filename=CUB_ARCHIVE_FILENAME,
                md5=CUB_MD5,
                remove_finished=remove_finished,
            )
            if _is_cub200_extracted(cub_dir):
                print(f"[CUB-200] Successfully extracted dataset to: {cub_dir}")
                return cub_dir
        except Exception as exc:
            last_err = exc
            print(f"[CUB-200] Warning: download attempt from {url} failed ({exc}).")

    raise RuntimeError(
        f"Failed to download and extract CUB-200-2011 into {root_path}."
    ) from last_err


def stratified_split_samples(
    samples: List[Tuple[Any, int]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[Any, int]], List[Tuple[Any, int]]]:
    """
    Split `(item, label)` samples into `(train_samples, val_samples)` using
    deterministic stratified sampling per class.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    rng = random.Random(seed)
    by_class: Dict[int, List[Tuple[Any, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    train_samples: List[Tuple[Any, int]] = []
    val_samples: List[Tuple[Any, int]] = []
    carry = 0.0

    for label in sorted(by_class.keys()):
        class_items = list(by_class[label])
        rng.shuffle(class_items)
        n = len(class_items)
        if n <= 1:
            train_samples.extend(class_items)
            continue

        exact_val = n * val_ratio + carry
        n_val = int(round(exact_val))
        n_val = max(1, min(n - 1, n_val))
        carry = exact_val - n_val

        val_samples.extend(class_items[:n_val])
        train_samples.extend(class_items[n_val:])

    return train_samples, val_samples


class CUB200Dataset(Dataset):
    """
    PyTorch Dataset for Caltech-UCSD Birds-200-2011 (CUB-200-2011).

    Reads official metadata files:
      - images.txt: `<image_id> <rel_path>`
      - image_class_labels.txt: `<image_id> <class_id>` (1..200 -> 0..199)
      - train_test_split.txt: `<image_id> <is_train>` (1 = train, 0 = test)
    """

    def __init__(
        self,
        root: Union[str, Path] = "data",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        samples: Optional[List[Tuple[Path, int]]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.cub_dir = self.root / CUB_EXTRACTED_FOLDER
        self.train = train
        self.transform = transform

        if samples is not None:
            self.samples: List[Tuple[Path, int]] = list(samples)
            self.class_names: List[str] = load_cub200_class_names(self.cub_dir)
            return

        if download:
            download_cub200(self.root)

        if not _is_cub200_extracted(self.cub_dir):
            raise FileNotFoundError(
                f"CUB-200-2011 dataset not found at '{self.cub_dir}'. "
                "Pass `download=True` or run `python src/data/cub200.py` first."
            )

        self.samples = self._load_metadata()
        self.class_names = load_cub200_class_names(self.cub_dir)

    def _load_metadata(self) -> List[Tuple[Path, int]]:
        images_file = self.cub_dir / "images.txt"
        labels_file = self.cub_dir / "image_class_labels.txt"
        split_file = self.cub_dir / "train_test_split.txt"

        img_paths: Dict[int, str] = {}
        with open(images_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_paths[int(parts[0])] = parts[1]

        labels: Dict[int, int] = {}
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    # Convert 1-indexed class IDs (1..200) to 0-indexed (0..199)
                    labels[int(parts[0])] = int(parts[1]) - 1

        target_split = 1 if self.train else 0
        samples: List[Tuple[Path, int]] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_id, is_train = int(parts[0]), int(parts[1])
                    if is_train == target_split:
                        full_path = self.cub_dir / "images" / img_paths[img_id]
                        samples.append((full_path, labels[img_id]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path, label = self.samples[index]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class HFCUB200Dataset(Dataset):
    """
    PyTorch Dataset wrapper for Hugging Face `Multimodal-Fatima/CUB_train` and `CUB_test`.

    Each record contains:
      - `image`: PIL bird photograph
      - `label`: class label (0..199 int or species string)
      - `description`: newline-separated string of 10 human-written captions
      - `file_name`: original CUB file name
      - `id`: row identifier

    Automatically splits and cleans the `description` column via `src/data/captions.py`.
    """

    def __init__(
        self,
        train: bool = True,
        transform: Optional[Callable] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        hf_dataset: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
        return_captions: bool = False,
        label_to_idx: Optional[Dict[str, int]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.train = train
        self.transform = transform
        self.return_captions = return_captions

        if hf_dataset is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Hugging Face `datasets` package is required for source='huggingface'. "
                    "Install it via `pip install datasets`."
                ) from exc

            repo_id = HF_CUB_TRAIN_REPO if train else HF_CUB_TEST_REPO
            split_name = "train" if train else "test"
            cache_str = str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
            hf_dataset = load_dataset(repo_id, split=split_name, cache_dir=cache_str)

        self.hf_dataset = hf_dataset
        self.indices: List[int] = (
            list(indices) if indices is not None else list(range(len(self.hf_dataset)))
        )

        self.class_names, self.label_to_idx = self._resolve_label_mapping(
            hf_dataset, label_to_idx=label_to_idx, class_names=class_names
        )
        # Lightweight `(dataset_idx, class_idx)` pairs matching `CUB200Dataset.samples` convention
        self.samples: List[Tuple[int, int]] = [
            (idx, self._extract_label_idx(self.hf_dataset[idx]["label"]))
            for idx in self.indices
        ]

    @staticmethod
    def _resolve_label_mapping(
        hf_dataset: Any,
        label_to_idx: Optional[Dict[str, int]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, int]]:
        if class_names is not None and label_to_idx is not None:
            return list(class_names), dict(label_to_idx)

        features = getattr(hf_dataset, "features", None)
        if isinstance(features, dict) and "label" in features:
            lbl_feat = features["label"]
            names = getattr(lbl_feat, "names", None)
            if names:
                formatted = [_format_species_name(str(n)) for n in names]
                mapping = {name: i for i, name in enumerate(formatted)}
                for i, raw_n in enumerate(names):
                    mapping[str(raw_n)] = i
                return formatted, mapping

        # Fallback: infer sorted unique labels from records
        raw_labels = [hf_dataset[i]["label"] for i in range(len(hf_dataset))]
        if raw_labels and isinstance(raw_labels[0], (int, np.integer)):
            max_lbl = int(max(raw_labels))
            total = max(200, max_lbl + 1)
            inferred_names = [f"Species_{i:03d}" for i in range(total)]
            return inferred_names, {n: i for i, n in enumerate(inferred_names)}

        unique_strs = sorted({_format_species_name(str(x)) for x in raw_labels})
        mapping = {name: i for i, name in enumerate(unique_strs)}
        return unique_strs, mapping

    def _extract_label_idx(self, raw_label: Any) -> int:
        if isinstance(raw_label, (int, np.integer)):
            return int(raw_label)
        cleaned = _format_species_name(str(raw_label))
        if cleaned in self.label_to_idx:
            return self.label_to_idx[cleaned]
        return self.label_to_idx[str(raw_label)]

    def get_cleaned_captions(self, index: int) -> List[str]:
        """Return cleaned list of captions for sample `index`."""
        raw_idx, _ = self.samples[index]
        record = self.hf_dataset[raw_idx]
        raw_caps, _ = split_raw_description(record.get("description", ""))
        cleaned: List[str] = []
        for cap in raw_caps:
            c, _ = clean_caption(cap)
            if len(c.split()) >= 2:
                cleaned.append(c)
        return cleaned

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        raw_idx, label = self.samples[index]
        record = self.hf_dataset[raw_idx]
        img = record["image"]
        if isinstance(img, (str, Path)):
            with Image.open(img) as opened:
                image = opened.convert("RGB")
        elif isinstance(img, Image.Image):
            image = img.convert("RGB")
        else:
            image = Image.fromarray(np.asarray(img)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_captions:
            return image, label, self.get_cleaned_captions(index)
        return image, label


def get_cub200_transforms(
    image_size: int = 224,
    augmentation: str = "basic",
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return `(train_transform, val_transform)` for CUB-200-2011 images.

    Augmentation modes:
      - `'basic'`: Resize(256/224*S) -> RandomCrop(S) -> RandomHorizontalFlip -> Normalize.
      - `'fine_grained'`: Domain-tailored for bird species recognition. Uses
        RandomResizedCrop(scale=(0.7, 1.0)), RandomHorizontalFlip, mild RandomRotation(15),
        and subtle illumination ColorJitter with near-zero hue shift (`hue=0.02`) so
        taxonomically critical plumage hues are never distorted.
      - `'strong'`: Adds mild GaussianBlur and RandomErasing (simulating branch/foliage occlusion)
        on top of `'fine_grained'`.
    """
    resize_dim = int(round(image_size * 256 / 224))
    aug_mode = str(augmentation).strip().lower()

    if aug_mode == "basic":
        train_ops: List[Any] = [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif aug_mode in ("fine_grained", "standard"):
        train_ops = [
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif aug_mode == "strong":
        train_ops = [
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.65, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.02,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), value="random"),
        ]
    else:
        raise ValueError(
            f"Unsupported augmentation strategy '{augmentation}'. "
            "Expected one of: ['basic', 'fine_grained', 'strong']."
        )

    train_transform = transforms.Compose(train_ops)
    val_transform = transforms.Compose(
        [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Invert channel normalization on a `(C, H, W)` image tensor and return an
    `(H, W, 3)` RGB float32 NumPy array clipped to `[0.0, 1.0]`.
    """
    img = tensor.detach().cpu().float()
    if img.ndim == 4:
        img = img[0]
    mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
    unnorm = (img * std_t + mean_t).clamp(0.0, 1.0)
    return unnorm.permute(1, 2, 0).numpy()


def get_dataset_class_names(dataset: Any, num_classes: int = 200) -> List[str]:
    """Extract ordered class names from a Dataset (or Subset) if available."""
    if hasattr(dataset, "class_names") and dataset.class_names:
        return list(dataset.class_names)
    if hasattr(dataset, "dataset"):
        return get_dataset_class_names(dataset.dataset, num_classes=num_classes)
    return [f"Species_{i:03d}" for i in range(num_classes)]


def create_dataloaders(
    cfg: Union[Dict[str, Any], DictConfig],
) -> Tuple[DataLoader, DataLoader]:
    """
    Create and return `(train_loader, val_loader)` for CUB-200-2011 from config
    using a stratified split (default 80/20) on the official train split.

    Supports both:
      - `data.source = 'local'` (official Caltech CUB_200_2011 directory)
      - `data.source = 'huggingface'` (`Multimodal-Fatima/CUB_train` with captions)
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, (dict, DictConfig)) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg, (dict, DictConfig)) else {}

    source = str(data_cfg.get("source", cfg.get("source", "local"))).strip().lower()
    data_dir = data_cfg.get("data_dir", cfg.get("data_dir", "data"))
    image_size = int(data_cfg.get("image_size", cfg.get("image_size", 224)))
    augmentation = str(data_cfg.get("augmentation", cfg.get("augmentation", "basic")))
    num_workers = int(data_cfg.get("num_workers", cfg.get("num_workers", 4)))
    download = bool(data_cfg.get("download", cfg.get("download", True)))
    val_split = float(data_cfg.get("val_split", cfg.get("val_split", 0.2)))
    seed = int(data_cfg.get("seed", cfg.get("seed", 42)))
    return_captions = bool(data_cfg.get("return_captions", False))
    batch_size = int(train_cfg.get("batch_size", cfg.get("batch_size", 32)))
    pin_memory = torch.cuda.is_available()

    train_transform, val_transform = get_cub200_transforms(
        image_size=image_size,
        augmentation=augmentation,
    )

    if source in ("hf", "huggingface"):
        full_hf = HFCUB200Dataset(
            train=True,
            transform=None,
            cache_dir=data_dir,
            return_captions=return_captions,
        )
        train_pairs, val_pairs = stratified_split_samples(
            full_hf.samples,
            val_ratio=val_split,
            seed=seed,
        )
        train_dataset: Dataset = HFCUB200Dataset(
            train=True,
            transform=train_transform,
            hf_dataset=full_hf.hf_dataset,
            indices=[idx for idx, _ in train_pairs],
            return_captions=return_captions,
            label_to_idx=full_hf.label_to_idx,
            class_names=full_hf.class_names,
        )
        val_dataset: Dataset = HFCUB200Dataset(
            train=True,
            transform=val_transform,
            hf_dataset=full_hf.hf_dataset,
            indices=[idx for idx, _ in val_pairs],
            return_captions=return_captions,
            label_to_idx=full_hf.label_to_idx,
            class_names=full_hf.class_names,
        )
    elif source == "local":
        full_train_dataset = CUB200Dataset(
            root=data_dir,
            train=True,
            transform=None,
            download=download,
        )
        train_samples, val_samples = stratified_split_samples(
            full_train_dataset.samples,
            val_ratio=val_split,
            seed=seed,
        )
        train_dataset = CUB200Dataset(
            root=data_dir,
            train=True,
            transform=train_transform,
            download=False,
            samples=train_samples,
        )
        val_dataset = CUB200Dataset(
            root=data_dir,
            train=True,
            transform=val_transform,
            download=False,
            samples=val_samples,
        )
    else:
        raise ValueError(
            f"Unsupported data.source '{source}'. Expected 'local' or 'huggingface'."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def create_test_dataloader(
    cfg: Union[Dict[str, Any], DictConfig],
    label_to_idx: Optional[Dict[str, int]] = None,
    class_names: Optional[List[str]] = None,
) -> DataLoader:
    """
    Create and return the held-out official test DataLoader (5,794 images) for
    final evaluation and fair benchmarking.
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, (dict, DictConfig)) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg, (dict, DictConfig)) else {}

    source = str(data_cfg.get("source", cfg.get("source", "local"))).strip().lower()
    data_dir = data_cfg.get("data_dir", cfg.get("data_dir", "data"))
    image_size = int(data_cfg.get("image_size", cfg.get("image_size", 224)))
    num_workers = int(data_cfg.get("num_workers", cfg.get("num_workers", 4)))
    download = bool(data_cfg.get("download", cfg.get("download", True)))
    return_captions = bool(data_cfg.get("return_captions", False))
    batch_size = int(train_cfg.get("batch_size", cfg.get("batch_size", 32)))
    pin_memory = torch.cuda.is_available()

    _, val_transform = get_cub200_transforms(image_size=image_size, augmentation="basic")

    if source in ("hf", "huggingface"):
        test_dataset: Dataset = HFCUB200Dataset(
            train=False,
            transform=val_transform,
            cache_dir=data_dir,
            return_captions=return_captions,
            label_to_idx=label_to_idx,
            class_names=class_names,
        )
    elif source == "local":
        test_dataset = CUB200Dataset(
            root=data_dir,
            train=False,
            transform=val_transform,
            download=download,
        )
    else:
        raise ValueError(
            f"Unsupported data.source '{source}'. Expected 'local' or 'huggingface'."
        )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract the CUB-200-2011 dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory where CUB_200_2011 will be downloaded and extracted (default: data).",
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Remove CUB_200_2011.tgz after extraction.",
    )
    args = parser.parse_args()
    download_cub200(root=args.data_dir, remove_finished=args.remove_archive)
