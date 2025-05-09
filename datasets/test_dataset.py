import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from utils import transforms


class TestDataset(Dataset):
    def __init__(self, config):
        data_folder = os.path.expanduser(config.test_data_folder)

        if not os.path.isdir(data_folder):
            raise RuntimeError(f'Test image directory: {data_folder} does not exist.')

        self.transform = AT.Compose([
            AT.Resize(960, 960),  # Match self.crop_size
            AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # Match training config if norm_mean/std were None
            AT.ToTensorV2(),
        ])

        self.images = []
        self.img_names = []
        for file_name in os.listdir(data_folder):
            full_path = os.path.join(data_folder, file_name)
            if os.path.isfile(full_path) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.images.append(full_path)
                self.img_names.append(file_name)

        print("Loaded {} images from {}".format(len(self.images), data_folder))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        #print('getitem image: ', image.min().item(), image.max().item())
        img_name = self.img_names[index]

        # Perform augmentation and normalization
        augmented = self.transform(image=image)
        image_aug = augmented['image']

        return image, image_aug, img_name