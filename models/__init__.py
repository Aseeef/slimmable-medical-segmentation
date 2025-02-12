import os, torch

from .unet import UNet
from .ducknet import DuckNet
from .resunet import ResUNet
from .model_registry import model_hub, aux_models


def get_model(config):
    if config.model == 'smp':   # Use segmentation models pytorch
        from .smp_wrapper import get_smp_model

        model = get_smp_model(config.encoder, config.decoder, config.encoder_weights, config.num_class)

    elif config.model in model_hub.keys():
        if config.model in aux_models:  # models support auxiliary heads
            model = model_hub[config.model](num_class=config.num_class, use_aux=config.use_aux)

        else:
            if config.use_aux:
                raise ValueError(f'Model {config.model} does not support auxiliary heads.\n')

            model = model_hub[config.model](num_class=config.num_class)

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


def get_teacher_model(config, device):
    if config.kd_training:
        if not os.path.isfile(config.teacher_ckpt):
            raise ValueError(f'Could not find teacher checkpoint at path {config.teacher_ckpt}.')   

        if config.teacher_model == 'smp':
            from .smp_wrapper import get_smp_model

            model = get_smp_model(config.teacher_encoder, config.teacher_decoder, None, config.num_class)

        else:
            raise NotImplementedError()

        teacher_ckpt = torch.load(config.teacher_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(teacher_ckpt['state_dict'])
        del teacher_ckpt

        model = model.to(device)    
        model.eval()
    else:
        model = None

    return model