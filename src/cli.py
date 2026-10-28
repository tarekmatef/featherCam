import hydra
from omegaconf import DictConfig, OmegaConf
from training.optimizer import OptimizerFactory
import torch
from torchvision.models import resnet18, ResNet18_Weights

@hydra.main(version_base=None, config_path="../config", config_name="main")
def feathercam(cfg: DictConfig):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    opt = OptimizerFactory.create_optimizer(model=model, cfg=cfg.training)
    print(opt)
    data = torch.rand(1,3,64,64)
    labels = torch.rand(1,1000)
    pred = model(data)
    loss = (labels - pred).sum()
    loss.backward()
    opt.step()



if __name__ == "__main__":
    feathercam()

