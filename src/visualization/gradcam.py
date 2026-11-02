"""
Pure PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
and Matplotlib explainability visualizations for correct and wrong predictions.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization", ICCV 2017.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

try:
    from src.data.cub200 import denormalize_tensor
except ImportError:
    from data.cub200 import denormalize_tensor


def resolve_target_layer(
    model: nn.Module,
    target_layer: Optional[Union[str, nn.Module]] = None,
) -> Tuple[nn.Module, str]:
    """
    Resolve the target convolutional activation layer in `model` for Grad-CAM.

    - If `target_layer` is an `nn.Module`, returns it directly.
    - If `target_layer` is a dot-separated submodule path (e.g. `'stage4.2'` or `'layer4.1'`),
      looks it up via `model.get_submodule`.
    - If `target_layer` is `None`, automatically locates the last `nn.Conv2d` in `model`
      and, if it sits inside an `nn.Sequential` immediately followed by `BatchNorm2d`
      and `ReLU` (prior to pooling), hooks that post-`ReLU` activation for maximum
      pre-pooling spatial resolution.
    """
    if isinstance(target_layer, nn.Module):
        for name, mod in model.named_modules():
            if mod is target_layer:
                return target_layer, name or target_layer.__class__.__name__
        return target_layer, target_layer.__class__.__name__

    if isinstance(target_layer, str) and target_layer.strip():
        clean_name = target_layer.strip()
        return model.get_submodule(clean_name), clean_name

    last_conv_name: Optional[str] = None
    last_conv_mod: Optional[nn.Module] = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last_conv_name = name
            last_conv_mod = mod

    if last_conv_mod is None or last_conv_name is None:
        raise ValueError(
            f"Could not auto-detect an `nn.Conv2d` layer in `{model.__class__.__name__}` for Grad-CAM. "
            "Use a convolutional backbone (e.g. `model_name=custom_cnn`) or pass `target_layer` explicitly."
        )

    # Check if parent is an nn.Sequential with post-BN ReLU immediately after this Conv2d
    if "." in last_conv_name:
        parent_path, child_key = last_conv_name.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        if isinstance(parent, nn.Sequential) and child_key.isdigit():
            idx = int(child_key)
            chosen_mod: nn.Module = last_conv_mod
            chosen_idx = idx
            for next_idx in range(idx + 1, len(parent)):
                sibling = parent[next_idx]
                if isinstance(sibling, (nn.BatchNorm2d, nn.ReLU)):
                    chosen_mod = sibling
                    chosen_idx = next_idx
                else:
                    break
            return chosen_mod, f"{parent_path}.{chosen_idx}"

    return last_conv_mod, last_conv_name


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN classifiers.

    Safely handles networks containing `nn.ReLU(inplace=True)` by cloning the hooked
    forward activation tensor and attaching a tensor-level autograd gradient hook.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[str, nn.Module]] = None,
    ) -> None:
        self.model = model
        self.target_layer, self.target_layer_name = resolve_target_layer(model, target_layer)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: torch.Tensor):
        if not isinstance(output, torch.Tensor):
            return output
        if not torch.is_grad_enabled() or not output.requires_grad:
            self.activations = output.detach()
            return output

        hooked = output.clone()
        self.activations = hooked.detach()
        hooked.register_hook(self._save_gradient)
        # Return an extra clone so downstream inplace ops (e.g. ReLU(inplace=True)) never mutate `hooked`
        return hooked.clone()

    def remove_hooks(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, Sequence[int], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Compute Grad-CAM spatial heatmaps for a single `(C, H, W)` or batched `(B, C, H, W)` input.

        Args:
            input_tensor: Normalized input image tensor.
            target_class: Class index (or per-sample sequence of class indices) to explain.
                Defaults to each sample's top-1 predicted class.

        Returns:
            Dictionary containing:
              - `'heatmaps'`: `(B, H, W)` float32 NumPy array normalized to `[0.0, 1.0]`
              - `'pred_classes'`: `(B,)` int64 NumPy array of top-1 predicted classes
              - `'pred_probs'`: `(B,)` float32 NumPy array of top-1 predicted confidences
              - `'target_classes'`: `(B,)` int64 NumPy array of explained target classes
              - `'target_probs'`: `(B,)` float32 NumPy array of probabilities for `target_classes`
              - `'logits'`: `(B, num_classes)` detached CPU tensor
        """
        was_training = self.model.training
        self.model.eval()

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = input_tensor.device

        x = input_tensor.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D image tensor for Grad-CAM, got shape {tuple(x.shape)}.")

        bsz, _, height, width = x.shape
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x_grad)
            probs = torch.softmax(logits, dim=1)
            pred_probs, pred_classes = probs.max(dim=1)

            if target_class is None:
                targets = pred_classes.clone()
            elif isinstance(target_class, int):
                targets = torch.full((bsz,), int(target_class), dtype=torch.long, device=device)
            else:
                targets = torch.as_tensor(target_class, dtype=torch.long, device=device).view(-1)
                if targets.numel() != bsz:
                    raise ValueError(
                        f"target_class length ({targets.numel()}) must match batch size ({bsz})."
                    )

            selected_scores = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            selected_scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                f"Grad-CAM hooks on layer '{self.target_layer_name}' did not capture 4D activations/gradients."
            )
        if self.activations.ndim != 4 or self.gradients.ndim != 4:
            raise ValueError(
                f"Target layer '{self.target_layer_name}' produced shape {tuple(self.activations.shape)}; "
                "Grad-CAM requires a 4D `(B, C, H', W')` spatial feature map."
            )

        # Global Average Pooling over spatial dimensions (H', W') -> channel weights alpha_k^c: (B, C, 1, 1)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # ReLU on linear combination of forward activation maps: (B, 1, H', W')
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # Bilinear upsampling to input resolution (H, W)
        cam_upsampled = F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Min-max normalize each sample heatmap independently to [0, 1]
        flat = cam_upsampled.view(bsz, -1)
        c_min = flat.min(dim=1)[0].view(bsz, 1, 1)
        c_max = flat.max(dim=1)[0].view(bsz, 1, 1)
        denom = (c_max - c_min).clamp_min(1e-8)
        cam_norm = torch.where(c_max > c_min, (cam_upsampled - c_min) / denom, torch.zeros_like(cam_upsampled))

        target_probs = probs.detach().gather(1, targets.unsqueeze(1)).squeeze(1)

        if was_training:
            self.model.train()

        return {
            "heatmaps": cam_norm.detach().cpu().float().numpy(),
            "pred_classes": pred_classes.detach().cpu().numpy(),
            "pred_probs": pred_probs.detach().cpu().float().numpy(),
            "target_classes": targets.detach().cpu().numpy(),
            "target_probs": target_probs.detach().cpu().float().numpy(),
            "logits": logits.detach().cpu(),
        }


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend a `[0, 1]` `(H, W)` Grad-CAM heatmap over a `[0, 1]` `(H, W, 3)` RGB image.

    Returns:
        `(colored_heatmap_rgb, blended_overlay_rgb)` both as float32 arrays in `[0, 1]`.
    """
    img = np.clip(np.asarray(image_rgb, dtype=np.float32), 0.0, 1.0)
    hm = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    cmap = matplotlib.colormaps.get_cmap(colormap) if hasattr(matplotlib, "colormaps") else mpl_cm.get_cmap(colormap)
    colored_hm = cmap(hm)[..., :3].astype(np.float32)
    blended = np.clip((1.0 - alpha) * img + alpha * colored_hm, 0.0, 1.0)
    return colored_hm, blended


def collect_correct_and_wrong_gradcam_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    num_correct: int = 4,
    num_wrong: int = 4,
    max_batches: int = 40,
) -> Dict[str, Any]:
    """
    Scan `dataloader` to collect both **correct** and **misclassified** predictions
    and compute Grad-CAM explanations (for misclassifications, computes heatmaps for
    BOTH the predicted wrong class and the ground-truth species for contrastive analysis).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.eval()

    correct_candidates: List[Tuple[torch.Tensor, int, int, float]] = []
    wrong_candidates: List[Tuple[torch.Tensor, int, int, float]] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            inputs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(inputs.size(0)):
                true_c = int(labels[i].item())
                pred_c = int(preds[i].item())
                conf = float(confs[i].item())
                img_cpu = inputs[i].detach().cpu()
                if pred_c == true_c:
                    correct_candidates.append((img_cpu, true_c, pred_c, conf))
                else:
                    wrong_candidates.append((img_cpu, true_c, pred_c, conf))

            if len(correct_candidates) >= num_correct * 3 and len(wrong_candidates) >= num_wrong * 3:
                break

    # Sort by highest confidence to inspect strong correct detections & confident errors
    correct_candidates.sort(key=lambda item: -item[3])
    wrong_candidates.sort(key=lambda item: -item[3])

    selected_correct = correct_candidates[:num_correct]
    selected_wrong = wrong_candidates[:num_wrong]

    correct_records: List[Dict[str, Any]] = []
    wrong_records: List[Dict[str, Any]] = []

    with GradCAM(model=model, target_layer=target_layer) as cam_engine:
        layer_name = cam_engine.target_layer_name

        for img_t, true_c, pred_c, _ in selected_correct:
            res = cam_engine.generate(img_t, target_class=pred_c)
            rgb = denormalize_tensor(img_t)
            hm = res["heatmaps"][0]
            hm_rgb, overlay = overlay_heatmap(rgb, hm)
            correct_records.append(
                {
                    "image_rgb": rgb,
                    "heatmap": hm,
                    "heatmap_rgb": hm_rgb,
                    "overlay_rgb": overlay,
                    "true_class": true_c,
                    "pred_class": int(res["pred_classes"][0]),
                    "pred_prob": float(res["pred_probs"][0]),
                    "is_correct": True,
                }
            )

        for img_t, true_c, pred_c, _ in selected_wrong:
            res_pred = cam_engine.generate(img_t, target_class=pred_c)
            res_true = cam_engine.generate(img_t, target_class=true_c)
            rgb = denormalize_tensor(img_t)
            hm_pred = res_pred["heatmaps"][0]
            hm_true = res_true["heatmaps"][0]
            hm_pred_rgb, overlay_pred = overlay_heatmap(rgb, hm_pred)
            _, overlay_true = overlay_heatmap(rgb, hm_true)
            wrong_records.append(
                {
                    "image_rgb": rgb,
                    "heatmap": hm_pred,
                    "heatmap_rgb": hm_pred_rgb,
                    "overlay_rgb": overlay_pred,
                    "true_class_overlay_rgb": overlay_true,
                    "true_class": true_c,
                    "pred_class": int(res_pred["pred_classes"][0]),
                    "pred_prob": float(res_pred["pred_probs"][0]),
                    "true_prob": float(res_true["target_probs"][0]),
                    "is_correct": False,
                }
            )

    return {
        "target_layer_name": layer_name,
        "correct": correct_records,
        "wrong": wrong_records,
    }


def plot_gradcam_predictions(
    records: Sequence[Dict[str, Any]],
    save_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    title: str = "Grad-CAM Explainability Analysis",
) -> Path:
    """
    Render a multi-row Matplotlib figure of Grad-CAM explanations.

    - For **correct** predictions: 3 columns `[Input Bird | Grad-CAM Activation Map | Overlay]`.
    - For **wrong** predictions (when `true_class_overlay_rgb` is present): 4 columns
      `[Input Bird | Pred (Wrong) Heatmap | Pred Class Overlay | Ground-Truth Class Overlay]`.
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, "No matching predictions found in evaluated split.", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title, fontweight="bold")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    has_contrastive = any("true_class_overlay_rgb" in r for r in records)
    n_cols = 4 if has_contrastive else 3
    n_rows = len(records)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.35 * n_rows))
    if n_rows == 1:
        axes_2d = np.expand_dims(axes, axis=0)
    else:
        axes_2d = axes

    def _cls_label(cid: int) -> str:
        if class_names and 0 <= cid < len(class_names):
            return class_names[cid][:26]
        return f"Species_{cid:03d}"

    for row_idx, rec in enumerate(records):
        true_lbl = _cls_label(int(rec["true_class"]))
        pred_lbl = _cls_label(int(rec["pred_class"]))
        conf_pct = float(rec["pred_prob"]) * 100.0
        is_ok = bool(rec.get("is_correct", rec["true_class"] == rec["pred_class"]))
        status_tag = "CORRECT" if is_ok else "MISCLASSIFIED"
        badge_color = "#2f855a" if is_ok else "#c53030"

        ax0 = axes_2d[row_idx, 0]
        ax0.imshow(rec["image_rgb"])
        ax0.set_title(f"[{status_tag}]\nTrue: {true_lbl}", fontsize=9, fontweight="bold", color=badge_color)
        ax0.axis("off")

        ax1 = axes_2d[row_idx, 1]
        ax1.imshow(rec["heatmap_rgb"])
        ax1.set_title(f"Grad-CAM Map\nPred: {pred_lbl} ({conf_pct:.1f}%)", fontsize=9)
        ax1.axis("off")

        ax2 = axes_2d[row_idx, 2]
        ax2.imshow(rec["overlay_rgb"])
        ax2.set_title("Overlay (Predicted Class)", fontsize=9, fontweight="bold")
        ax2.axis("off")

        if n_cols == 4:
            ax3 = axes_2d[row_idx, 3]
            if "true_class_overlay_rgb" in rec:
                true_pct = float(rec.get("true_prob", 0.0)) * 100.0
                ax3.imshow(rec["true_class_overlay_rgb"])
                ax3.set_title(f"Overlay (True Class)\nConf: {true_pct:.1f}%", fontsize=9)
            else:
                ax3.imshow(rec["overlay_rgb"])
                ax3.set_title("Overlay (Pred == True)", fontsize=9)
            ax3.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_and_plot_gradcam_analysis(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    class_names: Optional[Sequence[str]] = None,
    target_layer: Optional[Union[str, nn.Module]] = None,
    num_correct: int = 4,
    num_wrong: int = 4,
    model_name: str = "model",
) -> Dict[str, str]:
    """
    End-to-end Grad-CAM analysis pipeline: extracts correct & misclassified birds,
    renders Matplotlib figures for both groups, and logs figures to W&B if active.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected = collect_correct_and_wrong_gradcam_samples(
        model=model,
        dataloader=dataloader,
        device=device,
        target_layer=target_layer,
        num_correct=num_correct,
        num_wrong=num_wrong,
    )
    layer_name = collected["target_layer_name"]

    correct_path = plot_gradcam_predictions(
        collected["correct"],
        save_path=out_dir / f"gradcam_correct_{model_name}.png",
        class_names=class_names,
        title=f"Grad-CAM Correct Predictions — {model_name} (Layer: {layer_name})",
    )
    wrong_path = plot_gradcam_predictions(
        collected["wrong"],
        save_path=out_dir / f"gradcam_wrong_{model_name}.png",
        class_names=class_names,
        title=f"Grad-CAM Misclassified Predictions — {model_name} (Layer: {layer_name})",
    )

    if wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(
            {
                "explainability/gradcam_correct": wandb.Image(str(correct_path)),
                "explainability/gradcam_wrong": wandb.Image(str(wrong_path)),
            }
        )

    return {
        "target_layer": layer_name,
        "correct_figure": str(correct_path),
        "wrong_figure": str(wrong_path),
    }

