from .ducknet_34_config import DuckNet34Config
from .ducknet_17_config import DuckNet17Config
from .optuna_config import OptunaConfig

from .config_registry import config_hub


def get_config(name):
    if name in config_hub:
        dataset = config_hub[name]
    else:
        raise NotImplementedError('Unknown config!')

    return dataset


def list_available_configs():
    dataset_list = list(config_hub.keys())

    return dataset_list
