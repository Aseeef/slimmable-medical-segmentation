from .ducknet_34_config import DuckNet34Config
from .ducknet_32_config import DuckNet32Config
from .ducknet_17_config import DuckNet17Config
from .slim_ducknet_config import SlimDuckNetConfig
from .us_ducknet_config import USSlimDuckNetConfig
from .optuna_config import OptunaConfig
from .config_registry import config_hub
#Unfreeze Segmentation Head
from .bracs_ducknet_34_config import BracsDuckNet34Config
#Unfreeze Segmentation Head & Layer 1
from .bracs_ducknet_34_uf1_config import BracsDuckNet34_uf1_Config
from .bracs_ducknet_34_uf1_config_predict import BracsDuckNet34_uf1_Config_Predict
#Add 4 fully connected layers onto the ducknet segmentation head (seghead included)
from .bracs_ducknet_34_uf1_fchead_config import BracsDuckNet34_uf1_fchead_Config
#Arytenoids ducknet, 3 class classification.
from .arytenoids_ducknet_34_uf1_fchead_config import ArytenoidsDuckNet34_uf1_fchead_Config

def get_config(name):
    if name in config_hub:
        dataset = config_hub[name]
    else:
        raise NotImplementedError('Unknown config!')

    return dataset


def list_available_configs():
    dataset_list = list(config_hub.keys())

    return dataset_list
