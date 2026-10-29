from dotenv import load_dotenv
from hydra import compose, initialize
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.models.factory import ModelFactory
from src.training.optimizer import OptimizerFactory
from src.training.trainer import load_model_from_wandb, train


def test_wandb_training_and_model_reload():
    load_dotenv()

    with initialize(version_base=None, config_path="../config"):
        cfg = compose(
            config_name="main",
            overrides=[
                "model_name=mock",
                "training.epochs=2",
                "training.batch_size=4",
                "training.lr=0.01",
            ],
        )

    entity = cfg.tracker.entity
    project = cfg.tracker.project

    model = ModelFactory.create_model(cfg)
    optimizer = OptimizerFactory.create_optimizer(model, cfg.training)

    train_ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 200, (8,)))
    val_ds = TensorDataset(torch.randn(4, 3, 32, 32), torch.randint(0, 200, (4,)))
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

    run = wandb.init(
        entity=entity,
        project=project,
        name="pytest-wandb-weights-verification",
        job_type="test",
        config={
            "model_name": cfg.model_name,
            "optimizer": cfg.training.optimizer,
            "lr": cfg.training.lr,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
        },
    )
    run_url = run.url
    try:
        history = train(
            model=model,
            opt=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device="cpu",
        )
    finally:
        wandb.finish()

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2

    # Instantiate a fresh model with different random weights and load the trained weights from W&B
    reloaded_model = ModelFactory.create_model(cfg)
    assert not torch.allclose(model.fc.weight, reloaded_model.fc.weight)

    artifact_ref = f"{entity}/{project}/mock-weights:latest"
    load_model_from_wandb(reloaded_model, artifact_ref=artifact_ref, device="cpu")

    for (name_a, param_a), (name_b, param_b) in zip(
        model.named_parameters(), reloaded_model.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b), f"Mismatch in parameter {name_a}"

    model.eval()
    reloaded_model.eval()
    test_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(test_input), reloaded_model(test_input))

    print(f"W&B run recorded and model weights reloaded successfully! Run URL: {run_url}")
