import os


class BaseConfig:

    def __init__(self,):
        # Dataset
        self.dataset = None
        self.subset = None
        self.dataroot = None
        self.num_class = -1
        self.ignore_index = 255
        self.num_channel = None
        self.use_test_set = False

        # Model
        self.model = None
        self.encoder = None
        self.decoder = None
        self.encoder_weights = 'imagenet'
        self.base_channel = None

        # Training
        self.total_epoch = 200
        self.base_lr = 0.01
        self.train_bs = 16      # For each GPU
        self.use_aux = False
        self.aux_coef = None

        # Validating
        self.metrics = ['dice'] # The first one will be used as the main metric
        self.val_bs = 16        # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation
        self.val_img_stride = 1

        # Testing
        self.is_testing = False
        self.test_bs = 16
        self.test_data_folder = None
        self.colormap = 'random'
        self.colormap_path = None
        self.save_mask = True
        self.blend_prediction = True
        self.blend_alpha = 0.3

        # Loss
        self.loss_type = 'ce'
        self.class_weights = None
        self.ohem_thrs = 0.7
        self.reduction = 'mean'

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3

        # Optimizer
        self.optimizer_type = 'sgd'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        self.logger_name = None

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = False

        # Augmentation
        self.crop_size = 512
        self.crop_h = None
        self.crop_w = None
        self.scale = 1.0
        self.randscale = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.h_flip = 0.0
        self.v_flip = 0.0
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.affine_scale = None
        self.affine_translate = None
        self.affine_rotate = None
        self.affine_shear = None


        # DDP
        self.synBN = True
        self.destroy_ddp_process = True
        self.local_rank = int(os.getenv('LOCAL_RANK', -1))
        self.main_rank = self.local_rank in [-1, 0]

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = ''
        self.teacher_model = 'smp'
        self.teacher_encoder = None
        self.teacher_decoder = None
        self.kd_loss_type = 'kl_div'
        self.kd_loss_coefficient = 1.0
        self.kd_temperature = 4.0

        # Slim Size Multipliers
        self.slimmable_training = False
        self.nonuniform = False
        self.num_sample_training = 2
        self.slim_width_mult_list = None

        # The trainer to use
        self.trainer = 'segtrainer'

    def init_dependent_config(self):
        assert len(self.metrics) > 0

        if self.load_ckpt_path is None and not self.is_testing:
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        if self.crop_h is None:
            self.crop_h = self.crop_size

        if self.crop_w is None:
            self.crop_w = self.crop_size

        if self.dataset == 'polyp' or self.dataset == 'larynx_polyp':
            self.num_class = 2 if self.num_class == -1 else self.num_class
            self.num_channel = 3 if self.num_channel is None else self.num_channel