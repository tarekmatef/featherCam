import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.cub200 import CUB_EXTRACTED_FOLDER, create_dataloaders
from src.models.factory import ModelFactory
from src.training.optimizer import OptimizerFactory
from src.training.trainer import EarlyStopping, load_checkpoint, save_checkpoint, train

def test_mock_model_factory_and_backward():
    
    cfg = {
        "model_name": "mock",
        "training": {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-2}
    }

    model = ModelFactory.create_model(cfg)
    assert isinstance(model, torch.nn.Module)

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    prediction = model(dummy_input)
    assert prediction.shape == (batch_size, 200), f"Expected shape {(batch_size, 200)}, got {prediction.shape}"

    dummy_labels = torch.randint(0, 200, (batch_size,))
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(prediction, dummy_labels)
    loss.backward()

    # Ensure gradients were populated on leaf parameters
    assert model.fc.weight.grad is not None
    assert model.fc.bias.grad is not None

    optimizer = OptimizerFactory.create_optimizer(model, cfg["training"])
    optimizer.step()
    optimizer.zero_grad()
    assert model.fc.weight.grad is None


def test_batched_train_and_validate(tmp_path):
    cfg = {
        "model_name": "mock",
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 2,
            "batch_size": 4,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    model = ModelFactory.create_model(cfg)
    optimizer = OptimizerFactory.create_optimizer(model, cfg["training"])

    train_ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 200, (8,)))
    val_ds = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 200, (4,)))
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    history = train(model=model, opt=optimizer, train_loader=train_loader, val_loader=val_loader, cfg=cfg)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(0.0 <= acc <= 1.0 for acc in history["train_acc"])
    assert all(0.0 <= acc <= 1.0 for acc in history["val_acc"])


def test_cub200_dataloaders_with_train(tmp_path):
    cub_dir = tmp_path / CUB_EXTRACTED_FOLDER
    img_dir = cub_dir / "images" / "001.Black_footed_Albatross"
    img_dir.mkdir(parents=True)

    images_lines = []
    labels_lines = []
    split_lines = []

    # 10 train samples across 2 classes (5 per class -> 4 train + 1 val per class = 8 train, 2 val)
    # and 2 test samples (is_train=0) that must never appear in train or val
    for i in range(1, 13):
        rel_path = f"001.Black_footed_Albatross/img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 15, 100, 150)).save(cub_dir / "images" / rel_path)
        images_lines.append(f"{i} {rel_path}\n")
        class_id = 1 if i <= 5 else (2 if i <= 10 else 1)
        labels_lines.append(f"{i} {class_id}\n")
        split_lines.append(f"{i} {1 if i <= 10 else 0}\n")

    (cub_dir / "images.txt").write_text("".join(images_lines), encoding="utf-8")
    (cub_dir / "image_class_labels.txt").write_text("".join(labels_lines), encoding="utf-8")
    (cub_dir / "train_test_split.txt").write_text("".join(split_lines), encoding="utf-8")

    cfg = {
        "model_name": "mock",
        "data": {
            "data_dir": str(tmp_path),
            "image_size": 32,
            "num_workers": 0,
            "download": False,
            "val_split": 0.2,
            "seed": 42,
        },
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }

    train_loader, val_loader = create_dataloaders(cfg)
    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2

    train_labels = [lbl for _, lbl in train_loader.dataset.samples]
    val_labels = [lbl for _, lbl in val_loader.dataset.samples]
    assert train_labels.count(0) == 4 and train_labels.count(1) == 4
    assert val_labels.count(0) == 1 and val_labels.count(1) == 1

    train_paths = {p for p, _ in train_loader.dataset.samples}
    val_paths = {p for p, _ in val_loader.dataset.samples}
    assert train_paths.isdisjoint(val_paths)
    for test_idx in (11, 12):
        test_path = cub_dir / "images" / f"001.Black_footed_Albatross/img_{test_idx}.jpg"
        assert test_path not in train_paths
        assert test_path not in val_paths

    model = ModelFactory.create_model(cfg)
    optimizer = OptimizerFactory.create_optimizer(model, cfg["training"])
    history = train(model=model, opt=optimizer, train_loader=train_loader, val_loader=val_loader, cfg=cfg)
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1


def test_checkpoint_recovery_and_early_stopping(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    train_ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 200, (8,)))
    val_ds = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 200, (4,)))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    cfg_initial = {
        "model_name": "mock",
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 2,
            "batch_size": 4,
            "device": "cpu",
            "checkpoint_dir": str(ckpt_dir),
            "early_stopping": {"enabled": True, "monitor": "val_loss", "patience": 5},
        },
    }

    model = ModelFactory.create_model(cfg_initial)
    optimizer = OptimizerFactory.create_optimizer(model, cfg_initial["training"])
    history_part1 = train(model=model, opt=optimizer, train_loader=train_loader, val_loader=val_loader, cfg=cfg_initial)

    assert len(history_part1["train_loss"]) == 2
    assert (ckpt_dir / "latest_checkpoint.pt").exists()
    assert (ckpt_dir / "best_checkpoint.pt").exists()

    # Resume from latest checkpoint to epoch 4
    cfg_resume = {
        "model_name": "mock",
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 4,
            "batch_size": 4,
            "device": "cpu",
            "checkpoint_dir": str(ckpt_dir),
            "resume_from": True,
            "early_stopping": {"enabled": True, "monitor": "val_loss", "patience": 5},
        },
    }
    resumed_model = ModelFactory.create_model(cfg_resume)
    resumed_opt = OptimizerFactory.create_optimizer(resumed_model, cfg_resume["training"])
    history_part2 = train(
        model=resumed_model,
        opt=resumed_opt,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg_resume,
    )
    assert len(history_part2["train_loss"]) == 4
    assert history_part2["train_loss"][:2] == history_part1["train_loss"]

    # Verify early stopping triggers and restores best model weights
    es = EarlyStopping(patience=2, min_delta=0.0, monitor="val_loss", restore_best_weights=True)
    m = ModelFactory.create_model(cfg_initial)
    with torch.no_grad():
        m.fc.weight.fill_(1.0)
    assert es.step(1.5, model=m) is True  # Best score = 1.5

    with torch.no_grad():
        m.fc.weight.fill_(2.0)
    assert es.step(1.6, model=m) is False
    assert not es.early_stop

    with torch.no_grad():
        m.fc.weight.fill_(3.0)
    assert es.step(1.7, model=m) is False
    assert es.early_stop

    m.load_state_dict(es.best_state_dict)
    assert torch.all(m.fc.weight == 1.0)


if __name__ == "__main__":
    test_mock_model_factory_and_backward()
    test_batched_train_and_validate()
    print("Mock model, factory, backward pass, and optimizer tests passed successfully!")
