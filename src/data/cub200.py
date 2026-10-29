import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive

CUB_ARCHIVE_FILENAME = "CUB_200_2011.tgz"
CUB_EXTRACTED_FOLDER = "CUB_200_2011"
CUB_MD5 = "97eceeb196236b17998738112f37df78"
CUB_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
    "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz",
]

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


class CUB200Dataset(Dataset):
    """
    PyTorch Dataset for Caltech-UCSD Birds-200-2011 (CUB-200-2011).

    Reads official metadata files:
      - images.txt: `<image_id> <rel_path>`
      - image_class_labels.txt: `<image_id> <class_id>` (1..200 -> 0..199)
      - train_test_split.txt: `<image_id> <is_train>` (1 = train, 0 = validation/test)
    """

    def __init__(
        self,
        root: Union[str, Path] = "data",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.cub_dir = self.root / CUB_EXTRACTED_FOLDER
        self.train = train
        self.transform = transform

        if download:
            download_cub200(self.root)

        if not _is_cub200_extracted(self.cub_dir):
            raise FileNotFoundError(
                f"CUB-200-2011 dataset not found at '{self.cub_dir}'. "
                "Pass `download=True` or run `python src/data/cub200.py` first."
            )

        self.samples: List[Tuple[Path, int]] = self._load_metadata()

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


def get_cub200_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return `(train_transform, val_transform)` for CUB-200-2011 images."""
    resize_dim = int(image_size * 256 / 224)
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((resize_dim, resize_dim)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def create_dataloaders(
    cfg: Union[Dict[str, Any], DictConfig],
) -> Tuple[DataLoader, DataLoader]:
    """
    Create and return `(train_loader, val_loader)` for CUB-200-2011 from config.
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, (dict, DictConfig)) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg, (dict, DictConfig)) else {}

    data_dir = data_cfg.get("data_dir", cfg.get("data_dir", "data"))
    image_size = int(data_cfg.get("image_size", cfg.get("image_size", 224)))
    num_workers = int(data_cfg.get("num_workers", cfg.get("num_workers", 4)))
    download = bool(data_cfg.get("download", cfg.get("download", True)))
    batch_size = int(train_cfg.get("batch_size", cfg.get("batch_size", 32)))
    pin_memory = torch.cuda.is_available()

    train_transform, val_transform = get_cub200_transforms(image_size=image_size)

    train_dataset = CUB200Dataset(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=download,
    )
    val_dataset = CUB200Dataset(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=False,
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
