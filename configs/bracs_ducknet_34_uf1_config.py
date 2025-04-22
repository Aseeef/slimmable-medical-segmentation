import os.path

from .base_config import BaseConfig
from .config_registry import register_config


@register_config
class BracsDuckNet34_uf1_Config(BaseConfig):

    def __init__(self,):
        super().__init__()
        # Config name; used for save path
        self.save_dir = 'save/bracsducknet_34_uf1'

        # Dataset
        self.dataset = 'larynx_seg'
        self.subset = 'kvasir'
        self.data_root = r"/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/LarynxDataset/LarynxDataset"  #os.path.join('PolypDataset', 'LarynxDataset')
        #self.data_root = r'/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/medical-segmentation-pytorch/LarynxDataset/LarynxDataset'
        
        print(f'looking for data in {self.data_root}')

        self.use_test_set = True

        # Model
        self.model = 'bracsducknet_uf1'
        self.base_channel = 34
        self.num_class = 13

        # Training
        self.total_epoch = 250
        self.train_bs = 16  # this is PER GPU
        self.loss_type = 'modded_dice'
        self.optimizer_type = 'rmsprop'
        self.base_lr = 1e-5

        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'ducknet_uf1_trainer'

        # Augmentation
        #self.crop_size = 320
        self.randscale = None
        self.brightness = [0.6, 1.6]
        self.contrast = 0.2
        self.saturation = 0.1
        self.norm_mean = None
        self.norm_std = None
        self.affine_shear = (-22.5, 22)
        self.affine_rotate = (-30, 30)
        self.affine_translate = (-0.125, 0.125)
        self.affine_scale = (0.5, 1.5)

        # Trainer
        self.trainer = 'bracstrainer'
