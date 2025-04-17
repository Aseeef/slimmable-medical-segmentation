import os, torch

from .ducknet import DuckNet
from .slimmable.slimmable_ducknet import SlimmableDuckNet
from .resunet import ResUNet
from .resunetpp import ResUNetPP
from .unet import UNet
from .unetpp import UNetPP
from .model_registry import model_hub, aux_models, slimmable_models
#Segmentation Head Unfrozen
from .BRACSDuckNet import BRACSDuckNet
#Segmentation head and Convlayer 1 Unfrozen
from .BRACSDuckNet_UF1 import BRACSDuckNet_UF1
from .BRACSDuckNet_UF1_FCHead import BRACSDuckNet_UF1_FCHead
#3-class segmentation: left and right arytenoids.
from .ArytenoidsDuckNet_UF1_FCHead import ArytenoidsDuckNet_UF1_FCHead

def get_model(config):
    if config.model == 'smp':   # Use segmentation models pytorch
        from .smp_wrapper import get_smp_model

        model = get_smp_model(config.encoder, config.decoder, config.encoder_weights, config.num_class)

    elif config.model in model_hub.keys():
        if config.model in aux_models:  # models support auxiliary heads
            model = model_hub[config.model](num_class=config.num_class, base_channel=config.base_channel, use_aux=config.use_aux)

        elif config.model in slimmable_models:
            model = model_hub[config.model](slim_width_mult_list=config.slim_width_mult_list, num_class=config.num_class, base_channel=config.base_channel)

        else:
            if config.use_aux:
                raise ValueError(f'Model {config.model} does not support auxiliary heads.\n')

            model = model_hub[config.model](num_class=config.num_class, base_channel=config.base_channel)

    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model


def list_available_models():
    model_list = list(model_hub.keys())

    try:
        import segmentation_models_pytorch as smp
        model_list.append('smp')
    except:
        pass

    return model_list