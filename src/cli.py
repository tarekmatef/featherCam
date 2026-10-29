from pathlib import Path
import sys

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure `src` root is on sys.path whether invoked as `python src/cli.py` or `python -m src.cli`
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from models.factory import ModelFactory
from training.optimizer import OptimizerFactory
from training.trainer import train, validate


def review_config(cfg: DictConfig) -> None:
    """Display a structured summary of the active configuration before execution."""
    run_mode = cfg.get("run_mode", "train")
    model_name = cfg.get("model_name", "custom_cnn")
    train_cfg = cfg.get("training", {})
    tracker_cfg = cfg.get("tracker", {})
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = train_cfg.get("device", cfg.get("device", default_device))

    sep = "=" * 55
    print(f"\n{sep}")
    print("           FeatherCam Configuration Review")
    print(sep)
    print(f"  Run Mode       : {run_mode}")
    print(f"  Model          : {model_name}")
    print(f"  Device         : {device}")
    print("-" * 55)
    print("  Training Hyperparameters:")
    print(f"    Optimizer    : {train_cfg.get('optimizer', 'N/A')}")
    print(f"    Learning Rate: {train_cfg.get('lr', 'N/A')}")
    print(f"    Weight Decay : {train_cfg.get('weight_decay', 'N/A')}")
    print(f"    Betas        : {train_cfg.get('betas', 'N/A')}")
    print(f"    Eps          : {train_cfg.get('eps', 'N/A')}")
    print(f"    Epochs       : {train_cfg.get('epochs', 'N/A')}")
    print(f"    Batch Size   : {train_cfg.get('batch_size', 'N/A')}")
    print("-" * 55)
    print("  Experiment Tracker (W&B):")
    print(f"    Entity       : {tracker_cfg.get('entity', 'N/A')}")
    print(f"    Project      : {tracker_cfg.get('project', 'N/A')}")
    print(f"{sep}\n")


def confirm_execution(prompt: str = "Proceed with this configuration? [y/N]: ") -> bool:
    """Prompt the user for confirmation before starting the run."""
    try:
        response = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nExecution cancelled.")
        return False
    return response in ("y", "yes")


def _build_placeholder_loaders(cfg: DictConfig):
    """Build synthetic DataLoaders until the dataset pipeline in `src/data` is wired."""
    batch_size = int(cfg.training.get("batch_size", 32))
    train_ds = TensorDataset(
        torch.randn(batch_size * 2, 3, 224, 224),
        torch.randint(0, 200, (batch_size * 2,)),
    )
    val_ds = TensorDataset(
        torch.randn(batch_size, 3, 224, 224),
        torch.randint(0, 200, (batch_size,)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../config", config_name="main")
def feathercam(cfg: DictConfig):
    load_dotenv()
    OmegaConf.set_readonly(cfg, True)

    review_config(cfg)

    if not confirm_execution():
        print("Aborted by user.")
        return

    run_mode = str(cfg.get("run_mode", "train")).strip().lower()
    model = ModelFactory.create_model(cfg)

    if run_mode == "train":
        opt = OptimizerFactory.create_optimizer(model=model, cfg=cfg.training)
        train_loader, val_loader = _build_placeholder_loaders(cfg)
        train(
            model=model,
            opt=opt,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
        )
    elif run_mode in ("val", "validate", "eval"):
        _, val_loader = _build_placeholder_loaders(cfg)
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(cfg.training.get("device", default_device))
        model.to(device)
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
        )
        print(f"Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")
    else:
        raise ValueError(f"Unsupported run_mode: '{run_mode}'. Expected 'train' or 'validate'.")


if __name__ == "__main__":
    feathercam()


