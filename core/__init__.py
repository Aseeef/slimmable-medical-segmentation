from .base_trainer import BaseTrainer
from .bracs_trainer import SegTrainer
from .slim_seg_trainer import SlimmableSegTrainer
# from .jl_inference_runner import SlimmableSegInferenceRunner
from .loss import get_loss_fn
from .trainer_registry import trainer_hub


def get_trainer(config) -> BaseTrainer:

    # prints for clarity
    print(trainer_hub)
    print(config.trainer)
    
    if config.trainer in trainer_hub.keys():
        print('initializing!!')
        trainer = trainer_hub[config.trainer](config=config)
    else:
        raise NotImplementedError('Unsupported trainer!')

    return trainer
