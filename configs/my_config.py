from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Dataset
        self.dataset = 'polyp'
        self.subset = 'kvasir'
        self.data_root = r'/projectnb/ec523/projects/Team_A+/medical-segmentation-pytorch/PolypDataset/Kvasir-SEG'
        self.use_test_set = True

        # Model
        self.model = 'ducknet'
        self.base_channel = 32

        # Training
        self.total_epoch = 600
        self.train_bs = 3  # this is PER GPU
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
        self.randscale = [-0.5, 1.0]
        self.brightness = [0.6,1.6]
        self.contrast = 0.2
        self.saturation = 0.1
        self.h_flip = 0.5
        self.v_flip = 0.5
