import os, random, torch, json
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer


def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    keys = ['dataset', 'subset', 'num_class', 'model', 'encoder', 'decoder', 'loss_type', 
            'optimizer_type', 'lr_policy', 'total_epoch', 'train_bs', 'val_bs',  
            'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
            'DDP', 'synBN', 'use_ema']

    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)



def get_colormap(config):
    if config.colormap_path is not None and os.path.isfile(config.colormap_path):
        assert config.colormap_path.endswith('json')
        with open(config.colormap_path, 'r') as f:
            colormap_json = json.load(f)

        colormap = {int(k): tuple(v) for k, v in colormap_json.items()}

    else:
        if config.colormap == 'random':
            random_colors = np.random.randint(0, 256, size=(config.num_class, 3))
            colormap = {i: tuple(color) for i, color in enumerate(random_colors)}

        elif config.colormap == 'custom':
            raise NotImplementedError()

        else:
            raise ValueError(f'Unsupport colormap type: {config.colormap}.')

        # 🛠️ Fix starts here:
        colormap_json = {int(k): [int(x) for x in v] for k, v in colormap.items()}
        with open(f'{config.save_dir}/colormap.json', 'w') as f:
            json.dump(colormap_json, f, indent=1)

    colormap = [color for color in colormap.values()]

    if len(colormap) < config.num_class:
        raise ValueError('Length of colormap is smaller than the number of class.')
    else:
        return colormap[:config.num_class]

# def get_colormap(config):
#     if config.colormap_path is not None and os.path.isfile(config.colormap_path):
#         assert config.colormap_path.endswith('json')
#         with open(config.colormap_path, 'r') as f:
#             colormap_json = json.load(f)

#         colormap = {k: tuple(v) for k, v in colormap_json.items()}

#     else:
#         if config.colormap == 'random':
#             random_colors = np.random.randint(0, 256, size=(config.num_class, 3))
#             colormap = {i: tuple(color) for i, color in enumerate(random_colors)}

#         elif config.colormap == 'custom':
#             raise NotImplementedError()

#         else:
#             raise ValueError(f'Unsupport colormap type: {config.colormap}.')

#         colormap_json = {k: list(v) for k, v in colormap.items()}
#         with open(f'{config.save_dir}/colormap.json', 'w') as f:
#             json.dump(colormap_json, f, indent=1)

#     colormap = [color for color in colormap.values()]

#     if len(colormap) < config.num_class:
#         raise ValueError('Length of colormap is smaller than the number of class.')
#     else:
#         return colormap[:config.num_class]