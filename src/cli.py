import hydra
from omegaconf import DictConfig, OmegaConf

from models.factory import ModelFactory
from training.optimizer import OptimizerFactory


@hydra.main(version_base=None, config_path="../config", config_name="main")
def feathercam(cfg: DictConfig):
    OmegaConf.set_readonly(cfg, True)
    model = ModelFactory.create_model(cfg)
    opt = OptimizerFactory.create_optimizer(model=model, cfg=cfg.training)
    print(opt)
    print(model)

    

if __name__ == "__main__":
    feathercam()

