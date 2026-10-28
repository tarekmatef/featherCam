# src/training/optimizer.py
from typing import Any, Dict, Union

from omegaconf import DictConfig
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop


class OptimizerFactory:
    """
    Factory to instantiate PyTorch optimizers from a configuration dict/object.
    You can change the default optimizer settings from cli by appending training.{paramname}=value
    """


    _OPTIMIZERS = {
        "adam": Adam,
        "adamw": AdamW,
        "sgd": SGD,
        "rmsprop": RMSprop,
    }

    @classmethod
    def create_optimizer(
        cls, 
        model: nn.Module, 
        cfg: Union[Dict[str, Any], DictConfig]
    ) -> Optimizer:
        """
        Creates an optimizer for a model based on `cfg`.
        """

        opt_name = str(cfg.get('optimizer', "none")).strip().lower()
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 1e-2))

        # Filter parameters that require gradients
        params = [p for p in model.parameters() if p.requires_grad]

        if opt_name not in cls._OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: '{opt_name}'. "
                f"Supported options are: {list(cls._OPTIMIZERS.keys())}"
            )

        elif opt_name in ("adam", "adamw"):
            betas = tuple(cfg.get("betas", (0.9, 0.999)))
            eps = float(cfg.get("eps", 1e-8))
            optim_cls = cls._OPTIMIZERS[opt_name]
            return optim_cls(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        return cls._OPTIMIZERS[opt_name](params, lr=lr, weight_decay=weight_decay)
