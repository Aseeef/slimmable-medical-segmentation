from torch.utils.data import DataLoader

from .polyp import Polyp
from .larynx_polyp import Larynx_Seg
from .arytenoids import Arytenoids_Seg
from .dataset_registry import dataset_hub


def get_dataset(config, mode):
    if config.dataset in dataset_hub.keys():
        dataset = dataset_hub[config.dataset](config=config, mode=mode)
    else:
        raise NotImplementedError('Unsupported dataset!')

    return dataset


def get_loader(config, rank, mode, pin_memory=True, drop_last=True):
    dataset = get_dataset(config, mode)
    if mode == 'train':
        # Make sure train number is divisible by train batch size
        config.train_num = int(len(dataset) // config.train_bs * config.train_bs)
    elif mode == 'val':
        config.val_num = len(dataset)
    elif mode == 'test':
        config.test_num = len(dataset)

    shuffle = mode == 'train'
    if config.DDP:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=config.gpu_num, rank=rank, shuffle=shuffle)

        if mode == 'train':
            loader = DataLoader(dataset, batch_size=config.train_bs, shuffle=False, num_workers=config.num_workers, 
                                pin_memory=pin_memory, sampler=sampler, drop_last=drop_last)
        else:
            loader = DataLoader(dataset, batch_size=config.val_bs, shuffle=False, num_workers=config.num_workers, 
                                pin_memory=pin_memory, sampler=sampler)
    else:
        if mode == 'train':
            loader = DataLoader(dataset, batch_size=config.train_bs, shuffle=shuffle, num_workers=config.num_workers, drop_last=drop_last)
        else:
            loader = DataLoader(dataset, batch_size=config.val_bs, shuffle=shuffle, num_workers=config.num_workers)

    return loader


def get_test_loader(config): 
    from .test_dataset import TestDataset
    dataset = TestDataset(config)

    config.test_num = len(dataset)

    if config.DDP:
        raise NotImplementedError()

    else:
        test_loader = DataLoader(dataset, batch_size=config.test_bs, 
                                    shuffle=False, num_workers=config.num_workers)

    return test_loader


def list_available_datasets():
    dataset_list = list(dataset_hub.keys())

    return dataset_list
