from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch over batches and return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model over validation batches and return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(
    model: nn.Module,
    opt: Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    cfg: Optional[Union[Dict[str, Any], DictConfig]] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, list]:
    """
    Full training and validation loop over batches.

    Args:
        model: PyTorch model to train.
        opt: Optimizer instance.
        train_loader: DataLoader yielding (inputs, labels) training batches.
        val_loader: Optional DataLoader yielding (inputs, labels) validation batches.
        cfg: Hydra DictConfig or dict (looks under `cfg.training` or top-level `cfg` for `epochs`).
        criterion: Loss function (defaults to nn.CrossEntropyLoss()).
        device: Target device (defaults to CUDA if available, else CPU).

    Returns:
        Dictionary containing epoch history for `train_loss`, `train_acc`, `val_loss`, and `val_acc`.
    """
    cfg = cfg or {}
    train_cfg = cfg.get("training", cfg) if isinstance(cfg, (dict, DictConfig)) else {}
    epochs = int(train_cfg.get("epochs", cfg.get("epochs", 10)))

    if device is None:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_str = train_cfg.get("device", cfg.get("device", default_device))
        device = torch.device(device_str)
    else:
        device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.to(device)

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=opt,
            device=device,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/acc": train_acc,
        }

        if val_loader is not None:
            val_loss, val_acc = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            metrics["val/loss"] = val_loss
            metrics["val/acc"] = val_acc
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )

        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=epoch)

    if wandb is not None and getattr(wandb, "run", None) is not None:
        model_name = str(cfg.get("model_name", "model")).strip().lower() if isinstance(cfg, (dict, DictConfig)) else "model"
        artifact_name = f"{model_name}-weights"
        save_model_to_wandb(model=model, artifact_name=artifact_name, metadata={"epochs": epochs, **metrics})

    return history


def save_model_to_wandb(
    model: nn.Module,
    artifact_name: str = "model-weights",
    filename: str = "model_weights.pt",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save `model.state_dict()` and upload it as a W&B model artifact on the active run."""
    if wandb is None or getattr(wandb, "run", None) is None:
        raise RuntimeError("An active wandb.run is required to log model weights to W&B.")

    with tempfile.TemporaryDirectory() as tmpdir:
        weight_path = Path(tmpdir) / filename
        torch.save(model.state_dict(), weight_path)
        artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata or {})
        artifact.add_file(str(weight_path), name=filename)
        logged_artifact = wandb.run.log_artifact(artifact)
        logged_artifact.wait()
        return logged_artifact


def load_model_from_wandb(
    model: nn.Module,
    artifact_ref: str,
    filename: str = "model_weights.pt",
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """
    Download model weights from a W&B artifact and load them into `model`.

    Args:
        model: Uninitialized or freshly created nn.Module of matching architecture.
        artifact_ref: Full W&B artifact path, e.g. 'entity/project/artifact_name:version'.
        filename: Name of the weights file inside the artifact.
        device: Device to map loaded tensors onto.
    """
    if wandb is None:
        raise ImportError("wandb is not installed.")

    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="model")
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(artifact.download(root=tmpdir))
        state_dict = torch.load(artifact_dir / filename, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.to(device)
    return model


