import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.cub200 import CUB_EXTRACTED_FOLDER, create_dataloaders
from src.models.factory import ModelFactory
from src.training.optimizer import OptimizerFactory
from src.training.trainer import train

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


def test_batched_train_and_validate():
    cfg = {
        "model_name": "mock",
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 2,
            "batch_size": 4,
            "device": "cpu",
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

    for i in range(1, 7):
        rel_path = f"001.Black_footed_Albatross/img_{i}.jpg"
        Image.new("RGB", (64, 64), color=(i * 20, 100, 150)).save(cub_dir / "images" / rel_path)
        images_lines.append(f"{i} {rel_path}\n")
        labels_lines.append(f"{i} {(i % 200) + 1}\n")
        split_lines.append(f"{i} {1 if i <= 4 else 0}\n")

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
        },
        "training": {
            "optimizer": "adamw",
            "lr": 1e-2,
            "weight_decay": 1e-2,
            "epochs": 1,
            "batch_size": 2,
            "device": "cpu",
        },
    }

    train_loader, val_loader = create_dataloaders(cfg)
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2

    model = ModelFactory.create_model(cfg)
    optimizer = OptimizerFactory.create_optimizer(model, cfg["training"])
    history = train(model=model, opt=optimizer, train_loader=train_loader, val_loader=val_loader, cfg=cfg)
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1


if __name__ == "__main__":
    test_mock_model_factory_and_backward()
    test_batched_train_and_validate()
    print("Mock model, factory, backward pass, and optimizer tests passed successfully!")
