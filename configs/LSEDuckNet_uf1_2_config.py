import os.path

from .base_config import BaseConfig
from .config_registry import register_config


@register_config
class LSEDuckNet_uf1_2_config(BaseConfig):

    def __init__(self,):
        super().__init__()
        # Config name; used for save path
        self.save_dir = '/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/save/LSEducknet_34_uf1_2'
        self.save_ckpt = True
        self.load_ckpt = True
        self.load_ckpt_path = '/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/ducknet_34/best.pth'

        # Dataset
        self.dataset = 'polyp'
        self.subset = 'kvasir'
        self.data_root = '/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/LSEDataset/LSEDataset'
        self.use_test_set = True

        # Model
        self.model = 'lseducknet_uf1_2'
        self.base_channel = 34
        self.num_class = 2

        # Training
        self.total_epoch = 600
        self.train_bs = 4  # this is PER GPU
        self.loss_type = 'dice'
        self.optimizer_type = 'rmsprop'
        self.base_lr = 1e-5
        self.trainer='lsetrainer'
        self.resume_training = False 
        #Predicting
        #self.is_testing = True
        #self.save_mask = True
        #self.blend_prediction = True
        #self.test_data_folder = '/projectnb/ec523/projects/Team_A+/larynx_transfer_learning/github_branch/LSEDataset/LSEDataset/test'
    

        # Validating
        self.metrics = ['dice', 'iou']
        self.val_bs = 1

        # Training setting
        self.use_ema = False
        self.logger_name = 'LSEDucknet_Trainer'

        # Augmentation
        self.crop_size = 640
        self.randscale = None
        self.brightness = [0.6, 1.6]
        self.contrast = 0.2
        self.saturation = 0.1
        self.h_flip = 0.0
        self.v_flip = 0.0
        self.norm_mean = None
        self.norm_std = None
        self.affine_translate = (-0.1,0.1)
        self.affine_rotate = (-10,10)
        self.affine_shear = (0.9,1.1)
        self.affine_scale = (0.7, 1.2)
