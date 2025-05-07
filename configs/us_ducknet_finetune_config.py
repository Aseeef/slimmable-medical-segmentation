import os.path

from .base_config import BaseConfig, SlimmableTrainingType
from .config_registry import register_config


@register_config
class USSlimDuckNetFinetuneConfig(BaseConfig):

    def __init__(self,):
        super().__init__()
        # Config name; used for save path
        self.save_dir = 'save/us_ducknet_34_x2'

        # Dataset
        self.dataset = 'polyp'
        self.subset = 'kvasir'
        self.data_root = os.path.join('PolypDataset', 'Kvasir-SEG')
        self.use_test_set = True

        # Model
        self.model = 'usducknet'
        self.base_channel = 34

        # Training
        self.amp_training = False  # increases training speed by 7% in my tests
        self.trained_epochs = 600
        self.total_epoch = 800
        self.train_bs = 16  # this is PER GPU
        self.loss_type = 'dice'
        self.optimizer_type = 'adam'
        self.base_lr = 1e-4

        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'medseg_trainer'

        # Scheduler
        self.lr_policy = 'cos_anneal'

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

        # Slimmable Networks
        self.slimmable_training_type = SlimmableTrainingType.US_NET.value
        # note: if width multiplier result in round numbers, the decimal is truncated (so think math.floor)
        self.slim_width_mult_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        self.slim_width_range = [0.1, 1]
        self.inplace_distillation = True
        self.kd_loss_coefficient = 1.0
        self.kd_temperature = 4.0
        # the number of BATCHES to use for calibration (not the total number of training items)
        self.bn_calibration_batch_size = 3
        # how many width to sample for training
        self.us_num_training_samples = 6
        self.trainer = 'slimmablesegtrainer'