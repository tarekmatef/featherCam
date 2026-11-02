from pathlib import Path
import sys

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

# Ensure `src` root is on sys.path whether invoked as `python src/cli.py` or `python -m src.cli`
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.cub200 import create_dataloaders, create_test_dataloader
from models.factory import ModelFactory
from training.optimizer import OptimizerFactory
from training.trainer import evaluate_on_test, train, validate


def review_config(cfg: DictConfig) -> None:
    """Display a structured summary of the active configuration before execution."""
    run_mode = cfg.get("run_mode", "train")
    model_name = cfg.get("model_name", "custom_cnn")
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
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
    print("  Dataset (CUB-200-2011):")
    print(f"    Data Dir     : {data_cfg.get('data_dir', 'data')}")
    print(f"    Image Size   : {data_cfg.get('image_size', 224)}")
    print(f"    Num Workers  : {data_cfg.get('num_workers', 4)}")
    print(f"    Auto-Download: {data_cfg.get('download', True)}")
    print(f"    Val Split    : {data_cfg.get('val_split', 0.2)}")
    print(f"    Split Seed   : {data_cfg.get('seed', 42)}")
    print("-" * 55)
    print("  Training Hyperparameters:")
    print(f"    Optimizer    : {train_cfg.get('optimizer', 'N/A')}")
    print(f"    Head LR      : {train_cfg.get('lr', 'N/A')}")
    print(f"    Backbone LR  : {train_cfg.get('backbone_lr', 1e-5)}")
    print(f"    Weight Decay : {train_cfg.get('weight_decay', 'N/A')}")
    print(f"    Betas        : {train_cfg.get('betas', 'N/A')}")
    print(f"    Eps          : {train_cfg.get('eps', 'N/A')}")
    print(f"    Epochs       : {train_cfg.get('epochs', 'N/A')}")
    print(f"    Batch Size   : {train_cfg.get('batch_size', 'N/A')}")
    print(f"    Checkpoint   : {train_cfg.get('checkpoint_dir', 'outputs/checkpoints')}")
    print(f"    Resume From  : {train_cfg.get('resume_from', None)}")
    es_cfg = train_cfg.get("early_stopping", {}) or {}
    if es_cfg.get("enabled", True):
        print(
            f"    Early Stop   : monitor={es_cfg.get('monitor', 'val_loss')}, "
            f"patience={es_cfg.get('patience', 5)}, min_delta={es_cfg.get('min_delta', 0.0)}"
        )
    else:
        print("    Early Stop   : disabled")
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

    if run_mode in ("test", "evaluate"):
        test_loader = create_test_dataloader(cfg)
        evaluate_on_test(
            model=model,
            test_loader=test_loader,
            cfg=cfg,
            artifact_name=cfg.get("artifact_name"),
        )
        return

    train_loader, val_loader = create_dataloaders(cfg)

    if run_mode == "train":
        opt = OptimizerFactory.create_optimizer(model=model, cfg=cfg.training)
        wandb.init(
            entity=cfg.tracker.get("entity"),
            project=cfg.tracker.get("project"),
            job_type=run_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        try:
            train(
                model=model,
                opt=opt,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
            )
        finally:
            wandb.finish()
    elif run_mode in ("tune", "sweep"):
        sweep_cfg = {
            "method": "bayes",
            "metric": {"name": "val/loss", "goal": "minimize"},
            "parameters": {
                "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
                "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-1},
            },
        }
        entity = cfg.tracker.get("entity")
        project = cfg.tracker.get("project")
        sweep_id = wandb.sweep(sweep=sweep_cfg, entity=entity, project=project)

        def _sweep_trial() -> None:
            with wandb.init():
                trial_model = ModelFactory.create_model(cfg)
                trial_train_cfg = {**cfg.training, "lr": wandb.config.lr, "weight_decay": wandb.config.weight_decay}
                opt = OptimizerFactory.create_optimizer(model=trial_model, cfg=trial_train_cfg)
                train(model=trial_model, opt=opt, train_loader=train_loader, val_loader=val_loader, cfg=cfg)

        wandb.agent(sweep_id, function=_sweep_trial, entity=entity, project=project, count=int(cfg.training.get("sweep_count", 10)))
    elif run_mode in ("val", "validate", "eval"):
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
        raise ValueError(
            f"Unsupported run_mode: '{run_mode}'. Expected 'train', 'tune', 'validate', or 'test'."
        )


if __name__ == "__main__":
    feathercam()


