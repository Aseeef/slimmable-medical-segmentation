from torch.optim.lr_scheduler import OneCycleLR, StepLR, CosineAnnealingLR
from math import ceil


def get_scheduler(config, optimizer, trained_epochs=None):
    if config.DDP:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs/config.gpu_num)
    else:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs)
    config.total_itrs = int(config.total_epoch*config.iters_per_epoch)

    if config.lr_policy == 'cos_warmup':
        warmup_ratio = config.warmup_epochs / config.total_epoch
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=warmup_ratio)

    elif config.lr_policy == 'linear':
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=0., anneal_strategy='linear')

    elif config.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    elif config.lr_policy == 'cos_anneal':
        remaining_epochs = config.total_epoch - (0 if trained_epochs is None else trained_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

    else:
        raise NotImplementedError(f'Unsupported scheduler type: {config.lr_policy}')
    return scheduler