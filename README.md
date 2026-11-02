# featherCam

An end-to-end multimodal deep learning pipeline on **CUB-200-2011** featuring CNN fine-grained bird species classification with **Grad-CAM** interpretability, grayscale colorization, and autoregressive image captioning via a shared latent backbone. Built with **PyTorch**, **Hydra**, **Matplotlib**, and **Weights & Biases (W&B)**.

---

## 1. Environment Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

*(Optional)* Configure your W&B API key in a `.env` file (`WANDB_API_KEY=...`) or run `wandb login`.

---

## 2. CLI Commands (`src/cli.py`)

Append `confirm=false` to any command to run non-interactively without the `[y/N]` confirmation prompt.

### Task 1.1: Data Understanding, EDA & Caption Preprocessing
Runs full image + caption EDA (supporting Hugging Face `Multimodal-Fatima/CUB_train` & `CUB_test` or local Caltech `CUB_200_2011`), splits and cleans the `\n`-separated `description` column, checks for grayscale/corrupted/duplicate anomalies, and saves all Matplotlib figures and reports to `outputs/eda/` and `outputs/figures/eda/`:

```bash
python -m src.cli run_mode=eda eda.source=huggingface confirm=false
```

### Task 1.2: Train CNN Classifier (with Automatic Curves, Multi-Metric Eval & Grad-CAM)
```bash
# Train using local CUB-200-2011 archive:
python -m src.cli run_mode=train model_name=custom_cnn data.augmentation=fine_grained

# Or train directly from Hugging Face Multimodal-Fatima/CUB_train:
python -m src.cli run_mode=train model_name=custom_cnn data.source=huggingface
```

### Evaluation & Grad-CAM Explainability (Validation or Official Test Split)
Computes **Top-1 Accuracy, Top-5 Accuracy, Macro & Weighted Precision / Recall / F1-Score, 200x200 Confusion Matrix, Top Confused Species Pairs**, and **Grad-CAM heatmaps** on both **correct** and **misclassified** predictions:

```bash
# Evaluate best checkpoint on validation split:
python -m src.cli run_mode=validate evaluation.split=val evaluation.checkpoint_path=outputs/checkpoints/best_checkpoint.pt confirm=false

# Evaluate best checkpoint on official held-out test split (5,794 images):
python -m src.cli run_mode=validate evaluation.split=test evaluation.checkpoint_path=outputs/checkpoints/best_checkpoint.pt confirm=false

# Run standalone Grad-CAM visualization only:
python -m src.cli run_mode=gradcam evaluation.split=val evaluation.checkpoint_path=outputs/checkpoints/best_checkpoint.pt confirm=false
```

### Single-Image Inference + Grad-CAM Overlay
```bash
python -m src.cli run_mode=predict evaluation.image_path=path/to/bird.jpg evaluation.checkpoint_path=outputs/checkpoints/best_checkpoint.pt confirm=false
```

---

## 3. Running Unit Tests

```bash
pytest
```
