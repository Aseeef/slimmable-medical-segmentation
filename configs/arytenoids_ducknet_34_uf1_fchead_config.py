import os.path

from .base_config import BaseConfig
from .config_registry import register_config


@register_config
class ArytenoidsDuckNet34_uf1_fchead_Config(BaseConfig):

    def __init__(self,):
        super().__init__()
        # Config name; used for save path
        self.save_dir = '/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/save/arytenoidsducknet_34_uf1_fchead'
        self.save_example_path = r'/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/save/arytenoidsducknet_34_uf1_fchead'


        # Dataset
        self.dataset = 'larynx_seg'
        self.subset = 'kvasir'
        self.data_root = r"/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/ArytenoidsDataset/ArytenoidsDataset"
        print(f'looking for data in {self.data_root}')
        self.use_test_set = False

        # Model
        self.model = 'arytenoidsducknet_uf1_fchead'
        self.base_channel = 34
        self.num_class = 3

        # Training
        self.total_epoch = 600
        self.train_bs = 1  # this is PER GPU
        self.loss_type = 'ce_dice'
        self.optimizer_type = 'rmsprop'
        self.base_lr = 1e-5
        self.class_weights = (0.2, 1.8, 0)
        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'arytenoidsducknet_uf1_fchead_trainer'

        # Augmentation
        # Augmentation
        self.crop_size = 960
        self.crop_h = None
        self.crop_w = None
        self.scale = 1.0
        self.randscale = 0.0
        self.brightness = [0.5,1.5]
        self.contrast = 0.0
        self.saturation = 0.1
        self.h_flip = 0.0
        self.v_flip = 0.0
        self.norm_mean = None#[0.485, 0.456, 0.406]
        self.norm_std = None#[0.229, 0.224, 0.225]
        self.affine_scale = None
        self.affine_translate = (-0.1,0.1)
        self.affine_rotate = (-10,10)
        self.affine_shear = (0.9,1.1)

        # Trainer
        self.trainer = 'bracstrainer'
