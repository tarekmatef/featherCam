from omegaconf import DictConfig
import timm
from torch import nn
from typing import Union, Any, Dict
from .customcnn import FeatherCam, MockFeatherCam, FeatherNext


class ModelFactory:
    """
    Factory to instantiate PyTorch NN from configuration
    """

    _MODELS = {
        "custom_cnn": FeatherNext,
        "convnext": lambda: timm.create_model(
            "convnext_base.fb_in22k_ft_in1k_384",
            pretrained=True,
            num_classes=200,
        ),
        "mock": MockFeatherCam,
    }

    @classmethod
    def create_model(
        cls,
        cfg: Union[Dict[str, Any], DictConfig]
    ) -> nn:
        
        model_name = (cfg.get('model_name', "custom_cnn")).strip().lower()

        if model_name not in cls._MODELS:
            raise ValueError(
                    f"Unsupported model: '{model_name}'. "
                    f"Supported options are: {list(cls._MODELS.keys())}"
                )

        return cls._MODELS[model_name]()
            

