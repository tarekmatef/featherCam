from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    evaluate_classifier,
)
from src.models.customcnn import FeatherCam
from src.visualization.gradcam import (
    GradCAM,
    generate_and_plot_gradcam_analysis,
    resolve_target_layer,
)
from src.visualization.training_plots import (
    plot_confusion_matrix,
    plot_evaluation_diagnostics,
    plot_model_comparison,
    plot_training_curves,
)


def test_classification_metrics_and_confusion_matrix():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    logits = torch.tensor(
        [
            [5.0, 1.0, 0.0, -1.0],
            [1.0, 4.0, 0.5, -1.0],
            [0.2, 3.5, 0.1, -1.0],
            [0.1, 4.2, 0.3, -1.0],
            [0.0, 0.5, 3.8, -1.0],
            [3.0, 0.1, 2.5, -1.0],
        ]
    )

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=4)
    assert cm.shape == (4, 4)
    assert int(cm.sum()) == 6
    assert int(np.diag(cm).sum()) == 4  # 4/6 correct

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        logits=logits,
        num_classes=4,
        class_names=["Albatross", "Auklet", "Blackbird", "Bobolink"],
    )
    assert pytest.approx(metrics["accuracy"], rel=1e-5) == 4.0 / 6.0
    assert metrics["top5_accuracy"] == 1.0
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert 0.0 < metrics["macro_precision"] <= 1.0
    assert 0.0 < metrics["macro_recall"] <= 1.0
    assert len(metrics["top_confused_pairs"]) == 2


def test_evaluate_classifier_gradcam_and_plots(tmp_path: Path):
    torch.manual_seed(0)
    num_classes = 6
    class_names = [f"Bird_{i}" for i in range(num_classes)]
    model = FeatherCam(num_classes=num_classes)

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    # 1. Verify target layer resolution hooks stage4 post-ReLU (stage4.2)
    mod, layer_name = resolve_target_layer(model)
    assert layer_name == "stage4.2"
    assert isinstance(mod, torch.nn.ReLU)

    # 2. Verify Grad-CAM forward + backward pass works cleanly with inplace ReLU
    with GradCAM(model=model) as cam:
        out = cam.generate(images[:2])
        assert out["heatmaps"].shape == (2, 64, 64)
        assert float(out["heatmaps"].min()) >= 0.0
        assert float(out["heatmaps"].max()) <= 1.0 + 1e-6

    # 3. Verify full evaluator + report generation
    eval_dir = tmp_path / "eval"
    eval_res = evaluate_classifier(
        model=model,
        dataloader=loader,
        device="cpu",
        num_classes=num_classes,
        class_names=class_names,
        save_dir=eval_dir,
        model_name="custom_cnn",
        split_name="val",
    )
    assert (eval_dir / "metrics_val.json").is_file()
    assert (eval_dir / "evaluation_report_val.md").is_file()
    assert (eval_dir / "confusion_matrix_val.npy").is_file()

    # 4. Verify training curves, confusion matrix, diagnostics, comparison, and Grad-CAM plots
    figs_dir = tmp_path / "figs"
    p_curves = plot_training_curves(
        history={
            "train_loss": [2.1, 1.6, 1.2],
            "val_loss": [2.2, 1.7, 1.8],
            "train_acc": [0.25, 0.50, 0.72],
            "val_acc": [0.20, 0.48, 0.45],
        },
        save_path=figs_dir / "curves.png",
    )
    p_cm = plot_confusion_matrix(
        eval_res["confusion_matrix"],
        save_path=figs_dir / "cm.png",
        class_names=class_names,
    )
    p_diag = plot_evaluation_diagnostics(
        eval_res,
        save_path=figs_dir / "diag.png",
    )
    p_comp = plot_model_comparison(
        model_results={
            "FeatherCam (Scratch)": {"accuracy": 0.42, "top5_accuracy": 0.71, "macro_f1": 0.40},
            "ResNet-50 (Transfer)": {"accuracy": 0.81, "top5_accuracy": 0.95, "macro_f1": 0.80},
        },
        save_path=figs_dir / "comparison.png",
    )
    gradcam_figs = generate_and_plot_gradcam_analysis(
        model=model,
        dataloader=loader,
        save_dir=figs_dir,
        device="cpu",
        class_names=class_names,
        num_correct=2,
        num_wrong=2,
        model_name="custom_cnn",
    )

    for p in (
        p_curves,
        p_cm,
        p_diag,
        p_comp,
        Path(gradcam_figs["correct_figure"]),
        Path(gradcam_figs["wrong_figure"]),
    ):
        assert p.is_file() and p.stat().st_size > 0

