import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from .dataset_registry import register_dataset


@register_dataset
class Polyp(Dataset):
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
            if file_name.endswith('jpg'):
                img_path = os.path.join(img_dir, file_name)
                msk_path = os.path.join(msk_dir, file_name)

                if not os.path.isfile(msk_path):
                    raise RuntimeError(f'Mask file: {msk_path} not found.\n')

                self.images.append(img_path)
                self.masks.append(msk_path)

        if mode == 'train':
            self.transform = AT.Compose([
                AT.RandomScale(scale_limit=config.randscale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(0,0,0), mask_value=(0,0,0)),
                AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
                AT.HorizontalFlip(p=config.h_flip),
                AT.VerticalFlip(p=config.v_flip),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        elif mode in ['val', 'test']:
            self.transform = AT.Compose([
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),   
            ])

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
        mask = np.asarray(Image.open(self.masks[index]).convert('1')).astype(int)

        # Perform augmentation
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        return image, mask