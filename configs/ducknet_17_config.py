import os.path

from .base_config import BaseConfig
from .config_registry import register_config


@register_config
class DuckNet17Config(BaseConfig):

    def __init__(self,):
        super().__init__()
        # Change Save Dir for different configs!
        self.save_dir = 'save/ducknet_17'

        # Dataset
        self.dataset = 'polyp'
        self.subset = 'kvasir'
        self.data_root = os.path.join('PolypDataset', 'Kvasir-SEG')
        self.use_test_set = True

        # Model
        self.model = 'ducknet'
        self.base_channel = 17

        # Training
        self.total_epoch = 600
        self.train_bs = 16  # this is PER GPU
        self.loss_type = 'dice'
        self.optimizer_type = 'rmsprop'
        self.base_lr = 1e-4

        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'medseg_trainer'

        # Augmentation
        self.crop_size = 320
        self.randscale = None
        self.brightness = [0.6, 1.6]
        self.contrast = 0.2
        self.saturation = 0.1
        self.h_flip = 0.5
        self.v_flip = 0.5
        self.norm_mean = None
        self.norm_std = None
        self.affine_shear = (-22.5, 22)
        self.affine_rotate = (-180, 180)
        self.affine_translate = (-0.125, 0.125)
        self.affine_scale = (0.5, 1.5)
