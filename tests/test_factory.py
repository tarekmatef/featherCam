import pytest
import torch

from src.models.factory import ModelFactory
from src.training.optimizer import OptimizerFactory

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


if __name__ == "__main__":
    test_mock_model_factory_and_backward()
    print("Mock model, factory, backward pass, and optimizer tests passed successfully!")
