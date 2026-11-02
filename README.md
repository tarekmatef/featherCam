# FeatherCam

Fine-grained bird species classification on the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset (200 classes) built with PyTorch, Hydra, and Weights & Biases (W&B).

---

## Key Features
- **Leakage-Free Stratified Split**: Splits the official CUB-200-2011 training split (`is_train == 1`, 5,994 images) into **80% training (4,795 images)** and **20% validation (1,199 images)** using deterministic per-class stratified sampling (`seed: 42`). The official test split (`is_train == 0`, 5,794 images) is strictly held out for final evaluation.
- **Checkpointing & Error Recovery**: Saves atomic `latest_checkpoint.pt` and `best_checkpoint.pt` state files (model weights, optimizer state, epoch counter, metric history, and early stopping state) under `outputs/checkpoints/` so interrupted runs can be resumed seamlessly.
- **Early Stopping**: Monitors `val_loss` (or `val_acc`) with configurable patience (`default: 5`) and restores the best model weights before uploading the final artifact to W&B.
- **Gradient Monitoring**: Logs parameter gradients via `wandb.watch` during training to monitor for vanishing or exploding gradients.
- **Bayesian Hyperparameter Tuning**: Integrated W&B Sweeps (`run_mode=tune`) optimizing `lr` and `weight_decay`.
- **W&B Artifact Test Evaluation**: Evaluates any trained model artifact stored in W&B directly against the held-out CUB-200-2011 test split (`run_mode=test`).

---

## Project Structure

```text
featherCam/
├── config/
│   ├── main.yaml                 # Top-level Hydra configuration
│   ├── data/cub200.yaml          # Dataset & stratified split settings
│   ├── tracker/wandb.yaml        # W&B entity and project settings
│   └── training/training.yaml    # Optimizer, checkpointing, early stopping, and sweep settings
├── src/
│   ├── cli.py                    # Main CLI entrypoint (train, tune, validate, test)
│   ├── data/cub200.py            # CUB-200-2011 downloader, Dataset, and DataLoaders
│   ├── models/
│   │   ├── customcnn.py          # FeatherNext / FeatherCam / MockFeatherCam architectures
│   │   └── factory.py            # ModelFactory
│   └── training/
│       ├── optimizer.py          # OptimizerFactory (AdamW, Adam, SGD, RMSprop)
│       └── trainer.py            # Training loop, EarlyStopping, checkpointing, W&B model I/O, test evaluation
└── tests/
    ├── test_factory.py           # Unit tests for models, dataloaders, checkpointing, and evaluation
    └── test_wandb.py             # End-to-end W&B artifact upload/reload test
```

---

## 1. Environment Setup

Create a virtual or Conda environment and install the required dependencies:

```bash
conda create -n feathercam python=3.11 -y
conda activate feathercam
pip install -r requirements.txt
```

> **GPU Note**: `requirements.txt` defaults to the CPU PyTorch wheel index. For CUDA GPUs, update `--extra-index-url` in `requirements.txt` (e.g. `https://download.pytorch.org/whl/cu124`) before installing.

Configure your Weights & Biases API key in a `.env` file at the repository root (or run `wandb login`):

```bash
echo "WANDB_API_KEY=your_wandb_api_key_here" > .env
```

---

## 2. Dataset Preparation

The CUB-200-2011 dataset is downloaded and extracted automatically into `data/CUB_200_2011` when `data.download=true` (the default). You can also download it manually beforehand:

```bash
python src/data/cub200.py --data-dir data
```

---

## 3. Replicating Experiments

All commands use Hydra CLI overrides through `src/cli.py`. Before execution starts, the CLI displays a summary of the configuration and prompts for confirmation (`[y/N]`).

### A. Train a Model
Train `custom_cnn` with the default configuration (`AdamW`, `lr=1e-4`, `weight_decay=0.01`, `batch_size=32`, `epochs=10`, early stopping `patience=5`):

```bash
python src/cli.py
```

Override any hyperparameter from the command line:

```bash
python src/cli.py training.epochs=30 training.lr=0.0005 training.batch_size=64
```

At the end of training:
- Local checkpoints are saved to `outputs/checkpoints/latest_checkpoint.pt` and `outputs/checkpoints/best_checkpoint.pt`.
- The best model weights are uploaded to W&B as `<model_name>-weights` (e.g., `custom_cnn-weights:latest`).

### B. Resume / Continue Training from a Checkpoint
To recover from an interrupted run or continue training for more epochs from `outputs/checkpoints/latest_checkpoint.pt`:

```bash
python src/cli.py training.resume_from=true training.epochs=20
```

Or specify an explicit checkpoint file path (such as `best_checkpoint.pt` or a custom path):

```bash
python src/cli.py training.resume_from=outputs/checkpoints/best_checkpoint.pt training.epochs=25
```

### C. Hyperparameter Tuning (W&B Sweeps)
Run Bayesian hyperparameter tuning over `lr` (`[1e-5, 1e-2]`) and `weight_decay` (`[1e-5, 1e-1]`) to minimize `val/loss`:

```bash
python src/cli.py run_mode=tune training.sweep_count=10
```

### D. Evaluate a Model from W&B on the Held-Out Test Set
Evaluate the latest trained model artifact for `model_name` on the official CUB-200-2011 test split (`is_train == 0`):

```bash
python src/cli.py run_mode=test model_name=custom_cnn
```

To pick a specific model artifact name or version from W&B:

```bash
# By short artifact name (resolves to <entity>/<project>/<artifact_name>:latest)
python src/cli.py run_mode=test artifact_name=custom_cnn-weights:v2

# Or by full W&B artifact path
python src/cli.py run_mode=test artifact_name=feathercam/feathercam-cnn/custom_cnn-weights:latest
```

---

## 4. Running Unit Tests

Run the test suite with `pytest`:

```bash
pytest tests/test_factory.py
```
