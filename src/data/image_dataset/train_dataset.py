from typing import Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from torch import Tensor

from src.data.kitti.dataset import KITTIDataset

class TrainDataset(data.Dataset):
    
    def __init__(self, kitti_dataset: KITTIDataset, half_size: bool):
        self.kitti_dataset = kitti_dataset

        if half_size:
            self.crop_size = (172, 576)
        else:
            self.crop_size = (344, 1152)

        self.half_size = half_size

        # Interpolation parameters
        self.image_interpolation_mode = v2.InterpolationMode.BILINEAR
        self.depth_interpolation_mode = v2.InterpolationMode.NEAREST

        # Augmentation parameters
        self.angle_range = (-5., 5.)
        self.horizontal_flip_prob = 0.5
        self.color_jitter = v2.ColorJitter(0.1, 0.1, 0.1, 0.1)
        self.scale_range = (1., 1.5)

    def __len__(self) -> int:
        return len(self.kitti_dataset)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        np_image, _, np_gt_depth, np_velo_depth = self.kitti_dataset[index]
        
        image: torch.Tensor = torch.from_numpy(np_image).permute(2, 0, 1) # shape = (3, H, W)
        gt_depth: torch.Tensor = torch.from_numpy(np_gt_depth).unsqueeze(0) # shape = (1, H, W)
        velo_depth: torch.Tensor = torch.from_numpy(np_velo_depth).unsqueeze(0) # shape = (1, H, W)

        if self.half_size:
            # Half size
            size = image.shape[-2:]
            halved_size = (size[0]//2, size[1]//2)

            image = v2.functional.resize(image, size=halved_size, interpolation=self.image_interpolation_mode)
            gt_depth = v2.functional.resize(gt_depth, size=halved_size, interpolation=self.depth_interpolation_mode, antialias=False)
            velo_depth = v2.functional.resize(velo_depth, size=halved_size, interpolation=self.depth_interpolation_mode, antialias=False)

        # Color jitter
        image = self.color_jitter(image)

        # Random horizontal flip
        apply_horizontal_flip = (np.random.rand() < self.horizontal_flip_prob)
        if apply_horizontal_flip:
            image = v2.functional.horizontal_flip(image)
            gt_depth = v2.functional.horizontal_flip(gt_depth)
            velo_depth = v2.functional.horizontal_flip(velo_depth)

        # Random rotation
        angle = np.random.uniform(*self.angle_range)

        image = v2.functional.rotate(image, angle, interpolation=self.image_interpolation_mode)
        gt_depth = v2.functional.rotate(gt_depth, angle, interpolation=self.depth_interpolation_mode)
        velo_depth = v2.functional.rotate(velo_depth, angle, interpolation=self.depth_interpolation_mode)

        # Random scaling
        scale_factor = np.random.uniform(*self.scale_range)
        size = image.shape[-2:]
        
        image = v2.functional.resize(image, size=(int(size[0]*scale_factor), int(size[1]*scale_factor)), interpolation=self.image_interpolation_mode)
        gt_depth = v2.functional.resize(gt_depth, size=(int(size[0]*scale_factor), int(size[1]*scale_factor)), interpolation=self.depth_interpolation_mode)
        velo_depth = v2.functional.resize(velo_depth, size=(int(size[0]*scale_factor), int(size[1]*scale_factor)), interpolation=self.depth_interpolation_mode)
        
        # Adapt depth after scaling
        gt_depth /= scale_factor
        velo_depth /= scale_factor

        # Random crop
        crop_params = v2.RandomCrop.get_params(image, self.crop_size)

        image = v2.functional.crop(image, *crop_params)
        gt_depth = v2.functional.crop(gt_depth, *crop_params)
        velo_depth = v2.functional.crop(velo_depth, *crop_params)

        # Data type
        image = image.float() / 255 # shape = (3, H, W)
        gt_depth = gt_depth.float() # shape = (1, H, W)
        velo_depth = velo_depth.float() # shape = (1, H, W)

        return image, gt_depth, velo_depth