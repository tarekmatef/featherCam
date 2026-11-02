import copy
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


class EarlyStopping:
    """
    Monitor a validation metric and stop training when it stops improving.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        mode: Optional[str] = None,
        restore_best_weights: bool = True,
        best_score: Optional[float] = None,
        counter: int = 0,
    ) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor).strip()
        if mode is None:
            self.mode = "max" if "acc" in self.monitor else "min"
        else:
            self.mode = str(mode).strip().lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"Unsupported EarlyStopping mode '{self.mode}'. Expected 'min' or 'max'.")

        self.restore_best_weights = bool(restore_best_weights)
        self.best_score: Optional[float] = best_score
        self.counter: int = int(counter)
        self.early_stop: bool = False
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)

    def step(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Update early stopping state with `score`.
        Returns True if `score` is a new best, False otherwise.
        """
        if self.is_improvement(score):
            self.best_score = float(score)
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "mode": self.mode,
            "restore_best_weights": self.restore_best_weights,
            "best_score": self.best_score,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "best_state_dict": self.best_state_dict,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_score = state.get("best_score", self.best_score)
        self.counter = int(state.get("counter", self.counter))
        self.early_stop = bool(state.get("early_stop", False))
        if "best_state_dict" in state and state["best_state_dict"] is not None:
            self.best_state_dict = state["best_state_dict"]


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    history: Dict[str, list],
    early_stopping: Optional[EarlyStopping] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Atomically save full training state checkpoint to `filepath` for error recovery.
    """
    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    if early_stopping is not None:
        state["early_stopping"] = early_stopping.state_dict()
        state["best_score"] = early_stopping.best_score
    if extra_state:
        state.update(extra_state)

    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(target_path)
    return target_path


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load training state checkpoint from `filepath` into `model`, `optimizer`,
    and `early_stopping`. Returns the loaded checkpoint dictionary.
    """
    ckpt_path = Path(filepath)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at '{ckpt_path}'.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if early_stopping is not None and "early_stopping" in checkpoint:
        early_stopping.load_state_dict(checkpoint["early_stopping"])

    return checkpoint


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

    ckpt_dir_raw = train_cfg.get("checkpoint_dir", cfg.get("checkpoint_dir", "outputs/checkpoints"))
    checkpoint_dir: Optional[Path] = Path(ckpt_dir_raw) if ckpt_dir_raw else None
    resume_from = train_cfg.get("resume_from", cfg.get("resume_from", None))

    es_cfg = train_cfg.get("early_stopping", cfg.get("early_stopping", {})) or {}
    es_enabled = bool(es_cfg.get("enabled", True))
    early_stopping: Optional[EarlyStopping] = None
    if es_enabled:
        early_stopping = EarlyStopping(
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            monitor=str(es_cfg.get("monitor", "val_loss")),
            mode=es_cfg.get("mode", None),
            restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
        )

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    start_epoch = 1

    if resume_from:
        if isinstance(resume_from, bool):
            if checkpoint_dir is None:
                raise ValueError("Cannot resume with `resume_from=True` when `checkpoint_dir` is disabled.")
            resume_path = checkpoint_dir / "latest_checkpoint.pt"
        else:
            resume_path = Path(str(resume_from))
        ckpt = load_checkpoint(
            filepath=resume_path,
            model=model,
            optimizer=opt,
            early_stopping=early_stopping,
            device=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        loaded_history = ckpt.get("history")
        if isinstance(loaded_history, dict):
            for key in history:
                if key in loaded_history and isinstance(loaded_history[key], list):
                    history[key] = list(loaded_history[key])
        if early_stopping is not None:
            early_stopping.early_stop = False
        print(f"Resumed training state from '{resume_path}' at epoch {start_epoch}.")

    metrics: Dict[str, Any] = {}
    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    for epoch in range(start_epoch, epochs + 1):
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

        is_best = False
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

            if early_stopping is not None:
                monitored_value = val_acc if "acc" in early_stopping.monitor else val_loss
                is_best = early_stopping.step(monitored_value, model=model)
        else:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )

        if checkpoint_dir is not None:
            save_checkpoint(
                filepath=checkpoint_dir / "latest_checkpoint.pt",
                model=model,
                optimizer=opt,
                epoch=epoch,
                history=history,
                early_stopping=early_stopping,
            )
            if is_best:
                save_checkpoint(
                    filepath=checkpoint_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=opt,
                    epoch=epoch,
                    history=history,
                    early_stopping=early_stopping,
                )

        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=epoch)

        if early_stopping is not None and early_stopping.early_stop:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={early_stopping.patience}, best {early_stopping.monitor}={early_stopping.best_score:.4f})."
            )
            break

    if (
        early_stopping is not None
        and early_stopping.restore_best_weights
        and early_stopping.best_state_dict is not None
    ):
        model.load_state_dict(early_stopping.best_state_dict)
        model.to(device)

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


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    cfg: Optional[Union[Dict[str, Any], DictConfig]] = None,
    artifact_name: Optional[str] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[float, float]:
    """
    Load a model by name/artifact from W&B and evaluate it on `test_loader`.
    Returns `(test_loss, test_acc)`.
    """
    cfg = cfg or {}
    train_cfg = cfg.get("training", cfg) if isinstance(cfg, (dict, DictConfig)) else {}
    tracker_cfg = cfg.get("tracker", {}) if isinstance(cfg, (dict, DictConfig)) else {}

    if device is None:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_str = train_cfg.get("device", cfg.get("device", default_device))
        device = torch.device(device_str)
    else:
        device = torch.device(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    resume_from = train_cfg.get("resume_from", cfg.get("resume_from", None)) if isinstance(cfg, (dict, DictConfig)) else None
    if resume_from:
        ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        ckpt_path = (ckpt_dir / "best_checkpoint.pt") if isinstance(resume_from, bool) else Path(str(resume_from))
        load_checkpoint(filepath=ckpt_path, model=model, device=device)
        source_label = str(ckpt_path)
    else:
        if artifact_name is None:
            artifact_name = cfg.get("artifact_name") if isinstance(cfg, (dict, DictConfig)) else None

        if not artifact_name:
            model_name = str(cfg.get("model_name", "custom_cnn")).strip().lower() if isinstance(cfg, (dict, DictConfig)) else "custom_cnn"
            artifact_name = f"{model_name}-weights:latest"
        else:
            artifact_name = str(artifact_name).strip()
            if ":" not in artifact_name:
                if not artifact_name.endswith("-weights") and "/" not in artifact_name:
                    artifact_name = f"{artifact_name}-weights:latest"
                else:
                    artifact_name = f"{artifact_name}:latest"

        if "/" not in artifact_name:
            entity = tracker_cfg.get("entity", "feathercam")
            project = tracker_cfg.get("project", "feathercam-cnn")
            artifact_ref = f"{entity}/{project}/{artifact_name}"
        else:
            artifact_ref = artifact_name

        load_model_from_wandb(model=model, artifact_ref=artifact_ref, device=device)
        source_label = artifact_ref

    test_loss, test_acc = validate(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Test Evaluation ({source_label}) -> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    return test_loss, test_acc


