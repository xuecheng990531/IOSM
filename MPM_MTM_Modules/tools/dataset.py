import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from albumentations import Compose as ACompose
from torch.nn import functional as F
import albumentations as A
import random

class ToTensor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255.0  # RGB image to (C, H, W)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)  # Alpha to (1, H, W)

        # Validate trimap values
        if not np.all(np.isin(trimap, [0, 128, 255])):
            raise ValueError(f"Trimap contains invalid values: {np.unique(trimap)}")

        # Convert trimap to single channel [0, 0.5, 1]
        trimap_mapped = np.zeros_like(trimap, dtype=np.float32)
        trimap_mapped[trimap == 0] = 0.0    # 背景
        trimap_mapped[trimap == 128] = 0.5  # 未知
        trimap_mapped[trimap == 255] = 1.0  # 前景
        trimap_mapped[~np.isin(trimap, [0, 128, 255])] = 0.5  # 未知区域

        # 添加通道维度，形状从 (H, W) 变为 (1, H, W)
        trimap_mapped = np.expand_dims(trimap_mapped, axis=0)  # (1, H, W)

        sample['image'] = torch.from_numpy(image).sub_(self.mean).div_(self.std)  # [3, H, W]
        sample['alpha'] = torch.from_numpy(alpha)  # [1, H, W]
        sample['trimap'] = torch.from_numpy(trimap_mapped)  # [1, H, W]

        return sample

class MattingDataset(Dataset):
    def __init__(self, image_dir, trimap_dir, alpha_dir, mode='train', crop_size=512, seed=42):
        self.mode = mode
        self.crop_size = crop_size
        self.seed = seed

        if self.mode == 'train':
            self.image_dir = image_dir
            self.trimap_dir = trimap_dir
            self.alpha_dir = alpha_dir
        else:
            self.image_dir = image_dir.replace('train', 'test')
            self.trimap_dir = trimap_dir.replace('train', 'test')
            self.alpha_dir = alpha_dir.replace('train', 'test')

        self.image_names = sorted(os.listdir(self.image_dir))  # Sort for consistency

        # Define augmentation/resize pipelines
        self.train_augmentation = ACompose([
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.Resize(height=self.crop_size, width=self.crop_size, interpolation=cv2.INTER_CUBIC, p=1.0)
        ], additional_targets={'alpha': 'mask', 'trimap': 'mask'}) if mode == 'train' else None

        self.test_resize = ACompose([
            A.Resize(height=self.crop_size, width=self.crop_size, interpolation=cv2.INTER_CUBIC, p=1.0)
        ], additional_targets={'alpha': 'mask', 'trimap': 'mask'}) if mode == 'test' else None

        self.transform = ToTensor()
    

    def apply_augmentation(self, sample):
        if self.train_augmentation:
            h, w = sample['image'].shape[:2]
            alpha_h, alpha_w = sample['alpha'].shape
            trimap_h, trimap_w = sample['trimap'].shape

            if (h, w) != (alpha_h, alpha_w):
                sample['alpha'] = cv2.resize(sample['alpha'], (w, h), interpolation=cv2.INTER_NEAREST)
            if (h, w) != (trimap_h, trimap_w):
                sample['trimap'] = cv2.resize(sample['trimap'], (w, h), interpolation=cv2.INTER_NEAREST)

            data = {
                'image': sample['image'],
                'alpha': sample['alpha'],
                'trimap': sample['trimap']
            }
            augmented = self.train_augmentation(**data)
            sample['image'] = augmented['image']
            sample['alpha'] = augmented['alpha']
            sample['trimap'] = augmented['trimap']
        return sample

    def apply_test_resize(self, sample):
        if self.test_resize:
            h, w = sample['image'].shape[:2]
            alpha_h, alpha_w = sample['alpha'].shape
            trimap_h, trimap_w = sample['trimap'].shape

            if (h, w) != (alpha_h, alpha_w):
                sample['alpha'] = cv2.resize(sample['alpha'], (w, h), interpolation=cv2.INTER_NEAREST)
            if (h, w) != (trimap_h, trimap_w):
                sample['trimap'] = cv2.resize(sample['trimap'], (w, h), interpolation=cv2.INTER_NEAREST)

            data = {
                'image': sample['image'],
                'alpha': sample['alpha'],
                'trimap': sample['trimap']
            }
            resized = self.test_resize(**data)
            sample['image'] = resized['image']
            sample['alpha'] = resized['alpha']
            sample['trimap'] = resized['trimap']
        return sample

    def __len__(self):
        return len(self.image_names)

    def process_trimap(self, trimap):
        trimap = np.where(trimap < 64, 0, trimap)
        trimap = np.where((trimap >= 64) & (trimap < 192), 128, trimap)
        trimap = np.where(trimap >= 192, 255, trimap)
        return trimap

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        base_name, _ = os.path.splitext(img_name)
        image_path = os.path.join(self.image_dir, img_name)
        # print(f"Loading image: {image_path}")
        if 'aim500' in image_path:
            trimap_path = os.path.join(self.trimap_dir, base_name + '.jpg')
            alpha_path = os.path.join(self.alpha_dir, base_name + '.png')
        else:
            trimap_path = os.path.join(self.trimap_dir, base_name + '.jpg')
            alpha_path = os.path.join(self.alpha_dir, base_name + '.jpg')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        alpha = cv2.imread(alpha_path, 0).astype(np.float32) / 255.0
        alpha_shape = alpha.shape
        trimap = cv2.imread(trimap_path, 0)
        trimap = self.process_trimap(trimap)

        if image is None or alpha is None or trimap is None:
            raise ValueError(f"Failed to load files at index {idx}")

        sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'image_name': img_name}

        if self.mode == 'train':
            sample = self.apply_augmentation(sample)
        else:
            sample['shape'] = alpha_shape
            sample = self.apply_test_resize(sample)

        sample = self.transform(sample)
        return sample




if __name__ == "__main__":
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    from torchvision.utils import save_image
    torch.set_printoptions(profile="full", precision=3)

    # Training dataset
    # train_dataset = MattingDataset(
    #     image_dir="/icislab/volume1/lxc/composition-1k/train/merged",
    #     trimap_dir="/icislab/volume1/lxc/composition-1k/train/trimap",
    #     alpha_dir="/icislab/volume1/lxc/composition-1k/train/alpha",
    #     mode='train',
    #     seed=42
    # )
    train_dataset = MattingDataset(
        image_dir="data/aim500/train/img",
        trimap_dir="data/aim500/train/trimap",
        alpha_dir="data/aim500/train/alpha",
        mode='train',
        seed=42
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
    print(f"Training dataset size: {len(train_dataset)}")

    # Test dataset
    test_dataset = MattingDataset(
        image_dir="data/aim500/test/img",
        trimap_dir="data/aim500/test/trimap",
        alpha_dir="data/aim500/test/alpha",
        mode='test',
        seed=42
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Simulate multiple epochs
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")

        for batch_idx, batch in enumerate(train_dataloader):
            image, alpha, trimap = batch['image'], batch['alpha'], batch['trimap']
            print(f"Training batch {batch_idx} - trimap shape: {trimap.shape}, image shape: {image.shape}")
            print(trimap)
            break

        for batch_idx, batch in enumerate(test_dataloader):
            image, alpha, trimap, origing_shape = batch['image'], batch['alpha'], batch['trimap'], batch['shape']
            print(f"Test batch {batch_idx} - alpha shape: {alpha.shape}, trimap shape: {trimap.shape}")
            break