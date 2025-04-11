import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from .dataset_registry import register_dataset


@register_dataset
class Larynx_Seg(Dataset):
    def __init__(self, config, mode='train'):
        assert mode in ['train', 'val', 'test']
        mode_folder = mode if mode in ['train', 'test'] else 'validation'

        data_root = os.path.expanduser(config.data_root)
        data_folder = os.path.join(data_root, mode_folder)

        img_dir = os.path.join(data_folder, 'images')
        msk_dir = os.path.join(data_folder, 'masks')

        if not os.path.isdir(img_dir) or len(img_dir) == 0:
            raise RuntimeError('Image directory does not exist.\n')
        if not os.path.isdir(msk_dir) or len(msk_dir) == 0:
            raise RuntimeError('Mask directory does not exist.\n')

        self.images, self.masks = [], []
        for file_name in os.listdir(img_dir):
            #MY LINE OF CODE: MASK FILES ARE NOW .NPY FILES, NOT JPG FILES!
            mask_file_name = file_name.split('.jpg')[0]+'.npy'
            if file_name.endswith('jpg'):
                img_path = os.path.join(img_dir, file_name)
                msk_path = os.path.join(msk_dir, mask_file_name)

                if not os.path.isfile(msk_path):
                    raise RuntimeError(f'Mask file: {msk_path} not found.\n')

                self.images.append(img_path)
                self.masks.append(msk_path)

        self.transform = self.get_transform(config, mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        try:
            image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        except:
            # Use Opencv to load a tif image if PIL fails to load it
            import cv2
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = np.asarray(Image.open(self.masks[index]).convert('1')).astype(int)
        mask = np.load(self.masks[index])

        # Perform augmentation
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        return image, mask

    @staticmethod
    def get_transform(config, mode):
        """
        Constructs a transformation pipeline dynamically based on config, model, and mode.

        Args:
            config: Configuration object containing augmentation settings.
            mode: Dataset mode ('train', 'val', or 'test').

        Returns:
            Albumentations transformation pipeline.
        """
        transform_list = []

        if mode == 'train':
            # Apply augmentations only in training mode
            if config.randscale not in [None, 0]:
                transform_list.append(AT.RandomScale(scale_limit=config.randscale))

            transform_list.append(AT.PadIfNeeded(
                min_height=config.crop_h, min_width=config.crop_w,
                fill=(0, 0, 0), fill_mask=(0, 0, 0)
            ))

            transform_list.append(AT.RandomCrop(height=config.crop_h, width=config.crop_w))

            if any(x not in [None, 0] for x in [config.brightness, config.contrast, config.saturation]):
                transform_list.append(AT.ColorJitter(
                    brightness=config.brightness or 0,
                    contrast=config.contrast or 0,
                    saturation=config.saturation or 0
                ))

            if config.h_flip not in [None, 0]:
                transform_list.append(AT.HorizontalFlip(p=config.h_flip))

            if config.v_flip not in [None, 0]:
                transform_list.append(AT.VerticalFlip(p=config.v_flip))

            if any(x not in [None, 0] for x in
                   [config.affine_scale, config.affine_translate, config.affine_rotate, config.affine_shear]):
                transform_list.append(AT.Affine(
                    scale=config.affine_scale if config.affine_scale is not None else 1.0,
                    translate_percent=config.affine_translate if config.affine_translate is not None else 0,
                    rotate=config.affine_rotate if config.affine_rotate is not None else 0,
                    shear=config.affine_shear if config.affine_shear is not None else 0,
                ))

        elif mode in ['val', 'test']:
            # Validation and test transformations (no augmentations)
            transform_list.append(AT.PadIfNeeded(
                min_height=config.crop_h, min_width=config.crop_w,
                fill=(0, 0, 0), fill_mask=(0, 0, 0)
            ))
            transform_list.append(AT.CenterCrop(height=config.crop_h, width=config.crop_w))

        # Apply normalization if mean and std are available
        if config.norm_mean is not None and config.norm_std is not None:
            transform_list.append(AT.Normalize(mean=config.norm_mean, std=config.norm_std))

        # Convert to tensor (Always needed)
        transform_list.append(ToTensorV2())

        return AT.Compose(transform_list)
