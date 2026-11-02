from pathlib import Path
import random
import sys
from typing import Optional

from dotenv import load_dotenv
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import torch
import wandb

# Ensure both project root and `src` root are on sys.path whether invoked as
# `python src/cli.py` or `python -m src.cli`
_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.cub200 import (
    CUB_EXTRACTED_FOLDER,
    create_dataloaders,
    create_test_dataloader,
    denormalize_tensor,
    get_cub200_transforms,
    get_dataset_class_names,
    load_cub200_class_names,
)
from src.data.eda import run_eda
from src.evaluation.metrics import evaluate_classifier
from src.models.factory import ModelFactory
from src.training.optimizer import OptimizerFactory
from src.training.trainer import load_checkpoint, load_model_from_wandb, train
from src.visualization.gradcam import (
    GradCAM,
    generate_and_plot_gradcam_analysis,
    overlay_heatmap,
    plot_gradcam_predictions,
)
from src.visualization.training_plots import (
    plot_confusion_matrix,
    plot_evaluation_diagnostics,
    plot_training_curves,
)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def review_config(cfg: DictConfig) -> None:
    """Display a structured summary of the active configuration before execution."""
    run_mode = cfg.get("run_mode", "train")
    model_name = cfg.get("model_name", "custom_cnn")
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    eda_cfg = cfg.get("eda", {})
    tracker_cfg = cfg.get("tracker", {})
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = train_cfg.get("device", cfg.get("device", default_device))

    sep = "=" * 58
    print(f"\n{sep}")
    print("             FeatherCam Configuration Review")
    print(sep)
    print(f"  Run Mode       : {run_mode}")
    print(f"  Model          : {model_name}")
    print(f"  Device         : {device}")
    print(f"  Global Seed    : {cfg.get('seed', 42)}")
    print("-" * 58)
    print("  Dataset (CUB-200-2011):")
    print(f"    Source       : {data_cfg.get('source', 'local')}")
    print(f"    Data Dir     : {data_cfg.get('data_dir', 'data')}")
    print(f"    Image Size   : {data_cfg.get('image_size', 224)}")
    print(f"    Augmentation : {data_cfg.get('augmentation', 'fine_grained')}")
    print(f"    Num Workers  : {data_cfg.get('num_workers', 4)}")
    print(f"    Val Split    : {data_cfg.get('val_split', 0.2)}")
    if str(run_mode).lower() == "eda":
        print("-" * 58)
        print("  EDA Settings:")
        print(f"    EDA Source   : {eda_cfg.get('source', 'auto')}")
        print(f"    Output Dir   : {eda_cfg.get('output_dir', 'outputs/eda')}")
        print(f"    Figures Dir  : {eda_cfg.get('figures_dir', 'outputs/figures/eda')}")
    else:
        print("-" * 58)
        print("  Training / Evaluation:")
        print(f"    Optimizer    : {train_cfg.get('optimizer', 'N/A')}")
        print(f"    Learning Rate: {train_cfg.get('lr', 'N/A')}")
        print(f"    Weight Decay : {train_cfg.get('weight_decay', 'N/A')}")
        print(f"    Epochs       : {train_cfg.get('epochs', 'N/A')}")
        print(f"    Batch Size   : {train_cfg.get('batch_size', 'N/A')}")
        print(f"    Checkpoint   : {train_cfg.get('checkpoint_dir', 'outputs/checkpoints')}")
        print(f"    Eval Split   : {eval_cfg.get('split', 'val')}")
        print(f"    Run Grad-CAM : {eval_cfg.get('run_gradcam', True)}")
        es_cfg = train_cfg.get("early_stopping", {}) or {}
        if es_cfg.get("enabled", True):
            print(
                f"    Early Stop   : monitor={es_cfg.get('monitor', 'val_loss')}, "
                f"patience={es_cfg.get('patience', 5)}, min_delta={es_cfg.get('min_delta', 0.0)}"
            )
        else:
            print("    Early Stop   : disabled")
    print("-" * 58)
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


def _maybe_load_trained_weights(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> Optional[str]:
    """
    Load model weights for evaluation, Grad-CAM, or inference from:
      1. `evaluation.wandb_artifact` (if configured),
      2. `evaluation.checkpoint_path` or `training.resume_from`,
      3. `training.checkpoint_dir / best_checkpoint.pt` (or `latest_checkpoint.pt`).
    """
    eval_cfg = cfg.get("evaluation", {}) or {}
    train_cfg = cfg.get("training", {}) or {}

    wandb_ref = eval_cfg.get("wandb_artifact", None)
    if wandb_ref:
        load_model_from_wandb(model, artifact_ref=str(wandb_ref), device=device)
        print(f"[Weights] Loaded model weights from W&B artifact: {wandb_ref}")
        return str(wandb_ref)

    explicit_ckpt = eval_cfg.get("checkpoint_path", None) or train_cfg.get("resume_from", None)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
    candidate_path: Optional[Path] = None

    if explicit_ckpt:
        if isinstance(explicit_ckpt, bool):
            candidate_path = ckpt_dir / "best_checkpoint.pt"
            if not candidate_path.is_file():
                candidate_path = ckpt_dir / "latest_checkpoint.pt"
        else:
            candidate_path = Path(str(explicit_ckpt))
            if not candidate_path.is_file():
                raise FileNotFoundError(f"Configured checkpoint file not found: '{candidate_path}'")
    else:
        for fname in ("best_checkpoint.pt", "latest_checkpoint.pt"):
            cand = ckpt_dir / fname
            if cand.is_file():
                candidate_path = cand
                break

    if candidate_path is not None and candidate_path.is_file():
        payload = torch.load(candidate_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            load_checkpoint(filepath=candidate_path, model=model, device=device)
        else:
            model.load_state_dict(payload)
            model.to(device)
        print(f"[Weights] Loaded checkpoint from '{candidate_path}'.")
        return str(candidate_path)

    print(
        "[Warning] No trained checkpoint found; running with freshly initialized weights. "
        "Pass `evaluation.checkpoint_path=path/to/best_checkpoint.pt` to evaluate a trained model."
    )
    model.to(device)
    return None


def _run_evaluation_and_visualizations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    device: torch.device,
    split_name: str,
) -> dict:
    eval_cfg = cfg.get("evaluation", {}) or {}
    model_name = str(cfg.get("model_name", "custom_cnn")).strip().lower()
    out_dir = Path(eval_cfg.get("output_dir", "outputs/evaluation")) / model_name
    fig_dir = Path(eval_cfg.get("figures_dir", "outputs/figures/evaluation")) / model_name

    class_names = get_dataset_class_names(dataloader.dataset)
    results = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        save_dir=out_dir,
        model_name=model_name,
        split_name=split_name,
    )

    cm_fig = plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=fig_dir / f"confusion_matrix_{split_name}.png",
        class_names=class_names,
        title=f"Confusion Matrix — {model_name} ({split_name} split)",
    )
    diag_fig = plot_evaluation_diagnostics(
        results,
        save_path=fig_dir / f"evaluation_diagnostics_{split_name}.png",
        title=f"Species F1 Distribution & Top Confusions — {model_name} ({split_name} split)",
    )

    top5_str = (
        f" | Top-5 Acc: {results['top5_accuracy'] * 100:.2f}%"
        if results.get("top5_accuracy") is not None
        else ""
    )
    print(
        f"[{split_name.upper()} Eval] Loss: {results['loss']:.4f} | "
        f"Top-1 Acc: {results['accuracy'] * 100:.2f}%{top5_str} | "
        f"Macro F1: {results['macro_f1'] * 100:.2f}% | "
        f"Macro Prec: {results['macro_precision'] * 100:.2f}% | "
        f"Macro Rec: {results['macro_recall'] * 100:.2f}%"
    )
    print(f"[{split_name.upper()} Eval] Report  : {results.get('report_markdown_path')}")
    print(f"[{split_name.upper()} Eval] Figures : {cm_fig}, {diag_fig}")

    if wandb.run is not None:
        wandb.log(
            {
                f"{split_name}/confusion_matrix_plot": wandb.Image(str(cm_fig)),
                f"{split_name}/diagnostics_plot": wandb.Image(str(diag_fig)),
            }
        )

    if bool(eval_cfg.get("run_gradcam", True)):
        try:
            gradcam_figs = generate_and_plot_gradcam_analysis(
                model=model,
                dataloader=dataloader,
                save_dir=fig_dir,
                device=device,
                class_names=class_names,
                target_layer=eval_cfg.get("gradcam_target_layer", None),
                num_correct=int(eval_cfg.get("gradcam_num_correct", 4)),
                num_wrong=int(eval_cfg.get("gradcam_num_wrong", 4)),
                model_name=model_name,
            )
            print(
                f"[Grad-CAM] Layer '{gradcam_figs['target_layer']}' figures saved:\n"
                f"  - Correct : {gradcam_figs['correct_figure']}\n"
                f"  - Wrong   : {gradcam_figs['wrong_figure']}"
            )
        except ValueError as exc:
            print(f"[Grad-CAM] Skipped ({exc})")

    return results


def _run_single_image_prediction(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> None:
    eval_cfg = cfg.get("evaluation", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    img_path_raw = eval_cfg.get("image_path", None)
    if not img_path_raw:
        raise ValueError(
            "Single-image inference requires `evaluation.image_path=path/to/bird.jpg`."
        )
    img_path = Path(str(img_path_raw))
    if not img_path.is_file():
        raise FileNotFoundError(f"Input image not found at '{img_path}'.")

    image_size = int(data_cfg.get("image_size", 224))
    _, val_transform = get_cub200_transforms(image_size=image_size)
    with Image.open(img_path) as opened:
        pil_rgb = opened.convert("RGB")
    input_tensor = val_transform(pil_rgb)

    cub_dir = Path(data_cfg.get("data_dir", "data")) / CUB_EXTRACTED_FOLDER
    class_names = load_cub200_class_names(cub_dir)

    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_ids = probs.topk(min(5, probs.numel()))

    print(f"\nPredictions for '{img_path.name}':")
    for rank, (cid, p) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), start=1):
        species = class_names[cid] if cid < len(class_names) else f"Species_{cid:03d}"
        print(f"  {rank}. {species} (class {cid}) — {p * 100:.2f}%")

    pred_cid = int(top_ids[0].item())
    pred_prob = float(top_probs[0].item())
    fig_dir = Path(eval_cfg.get("figures_dir", "outputs/figures/evaluation")) / "predictions"
    try:
        with GradCAM(model=model, target_layer=eval_cfg.get("gradcam_target_layer", None)) as cam:
            cam_res = cam.generate(input_tensor)
            rgb = denormalize_tensor(input_tensor)
            hm = cam_res["heatmaps"][0]
            hm_rgb, overlay = overlay_heatmap(rgb, hm)
            out_fig = plot_gradcam_predictions(
                [
                    {
                        "image_rgb": rgb,
                        "heatmap": hm,
                        "heatmap_rgb": hm_rgb,
                        "overlay_rgb": overlay,
                        "true_class": pred_cid,
                        "pred_class": pred_cid,
                        "pred_prob": pred_prob,
                        "is_correct": True,
                    }
                ],
                save_path=fig_dir / f"gradcam_predict_{img_path.stem}.png",
                class_names=class_names,
                title=f"Prediction & Grad-CAM Explanation — {img_path.name}",
            )
            print(f"[Grad-CAM] Saved explanation overlay to: {out_fig}")
    except ValueError as exc:
        print(f"[Grad-CAM] Skipped ({exc})")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def feathercam(cfg: DictConfig):
    load_dotenv()
    OmegaConf.set_readonly(cfg, True)

    seed_everything(int(cfg.get("seed", 42)))
    review_config(cfg)

    if bool(cfg.get("confirm", True)) and not confirm_execution():
        print("Aborted by user.")
        return

    run_mode = str(cfg.get("run_mode", "train")).strip().lower()

    if run_mode == "eda":
        run_eda(cfg)
        return

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(cfg.training.get("device", cfg.get("device", default_device)))
    model_name = str(cfg.get("model_name", "custom_cnn")).strip().lower()
    model = ModelFactory.create_model(cfg)

    if run_mode == "predict":
        _maybe_load_trained_weights(model=model, cfg=cfg, device=device)
        _run_single_image_prediction(model=model, cfg=cfg, device=device)
        return

    eval_cfg = cfg.get("evaluation", {}) or {}
    eval_split = str(eval_cfg.get("split", "val")).strip().lower()

    if run_mode == "train":
        train_loader, val_loader = create_dataloaders(cfg)
        opt = OptimizerFactory.create_optimizer(model=model, cfg=cfg.training)
        wandb.init(
            entity=cfg.tracker.get("entity"),
            project=cfg.tracker.get("project"),
            job_type=run_mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        try:
            history = train(
                model=model,
                opt=opt,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg,
            )
            curves_path = plot_training_curves(
                history=history,
                save_path=Path("outputs/figures/training") / f"{model_name}_training_curves.png",
                title=f"Training vs. Validation Curves — {model_name}",
            )
            print(f"[Plots] Saved training curves to: {curves_path}")
            if wandb.run is not None:
                wandb.log({"training/curves_plot": wandb.Image(str(curves_path))})

            _run_evaluation_and_visualizations(
                model=model,
                dataloader=val_loader,
                cfg=cfg,
                device=device,
                split_name="val",
            )
        finally:
            wandb.finish()

    elif run_mode in ("val", "validate", "eval"):
        _maybe_load_trained_weights(model=model, cfg=cfg, device=device)
        if eval_split == "test":
            target_loader = create_test_dataloader(cfg)
        else:
            _, target_loader = create_dataloaders(cfg)
        _run_evaluation_and_visualizations(
            model=model,
            dataloader=target_loader,
            cfg=cfg,
            device=device,
            split_name=eval_split,
        )

    elif run_mode == "gradcam":
        _maybe_load_trained_weights(model=model, cfg=cfg, device=device)
        if eval_split == "test":
            target_loader = create_test_dataloader(cfg)
        else:
            _, target_loader = create_dataloaders(cfg)
        class_names = get_dataset_class_names(target_loader.dataset)
        fig_dir = Path(eval_cfg.get("figures_dir", "outputs/figures/evaluation")) / model_name
        gradcam_figs = generate_and_plot_gradcam_analysis(
            model=model,
            dataloader=target_loader,
            save_dir=fig_dir,
            device=device,
            class_names=class_names,
            target_layer=eval_cfg.get("gradcam_target_layer", None),
            num_correct=int(eval_cfg.get("gradcam_num_correct", 4)),
            num_wrong=int(eval_cfg.get("gradcam_num_wrong", 4)),
            model_name=model_name,
        )
        print(
            f"[Grad-CAM] Layer '{gradcam_figs['target_layer']}' figures saved:\n"
            f"  - Correct : {gradcam_figs['correct_figure']}\n"
            f"  - Wrong   : {gradcam_figs['wrong_figure']}"
        )

    else:
        raise ValueError(
            f"Unsupported run_mode: '{run_mode}'. "
            "Expected one of: 'train', 'validate', 'eda', 'gradcam', 'predict'."
        )


if __name__ == "__main__":
    feathercam()
