from typing import Optional

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as v2
from torch import Tensor

from src.data.kitti.dataset import KITTIDataset

class ValDataset(data.Dataset):
    
    def __init__(self, kitti_dataset: KITTIDataset, half_size: bool):
        self.kitti_dataset = kitti_dataset

        if half_size:
            self.crop_size = (172, 576)
        else:
            self.crop_size = (344, 1152)

        self.half_size = half_size
        self.center_crop = v2.CenterCrop(self.crop_size)

        # Interpolation parameters
        self.image_interpolation_mode = v2.InterpolationMode.BILINEAR
        self.depth_interpolation_mode = v2.InterpolationMode.NEAREST

    def __len__(self) -> int:
        return len(self.kitti_dataset)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        np_image, _, np_gt_depth, np_velo_depth = self.kitti_dataset[index]
        
        image: torch.Tensor = torch.from_numpy(np_image).permute(2, 0, 1) # shape = (3, H, W)
        gt_depth: torch.Tensor = torch.from_numpy(np_gt_depth).unsqueeze(0) # shape = (1, H, W)
        velo_depth: torch.Tensor = torch.from_numpy(np_velo_depth).unsqueeze(0) # shape = (1, H, W)
        
        #if self.half_size:
        #    # Half size (for gpu memory saving)
        #    size = image.shape[-2:]
        #    halved_size = (size[0]//2, size[1]//2)

        image = v2.functional.resize(image, size=self.crop_size, interpolation=self.image_interpolation_mode)
        gt_depth = v2.functional.resize(gt_depth, size=self.crop_size, interpolation=self.depth_interpolation_mode, antialias=False)
        velo_depth = v2.functional.resize(velo_depth, size=self.crop_size, interpolation=self.depth_interpolation_mode, antialias=False)

        #image = self.center_crop(image)
        #gt_depth = self.center_crop(gt_depth)
        #velo_depth = self.center_crop(velo_depth)

        # Data type
        image = image.float() / 255 # shape = (3, H, W)
        gt_depth = gt_depth.float() # shape = (1, H, W)
        velo_depth = velo_depth.float() # shape = (1, H, W)

        return image, gt_depth, velo_depth