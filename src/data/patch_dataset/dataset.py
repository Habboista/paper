import random
import os
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import skimage
import skimage.transform as T
import torch
import torch.utils.data as data
from tqdm import tqdm

from src.data.kitti.camera import Camera
from src.data.kitti.dataset import KITTIDataset
from src.data.kitti.utils import cloud2depth
from src.data.kitti.utils import depth2cloud
from src.data.kitti.utils import depth_np2torch
from src.data.kitti.utils import image_np2torch
from src.data.patch_dataset.sampling_strategies import corner_sample
from src.data.patch_dataset.sampling_strategies import edge_sample
from src.data.patch_dataset.sampling_strategies import grid_sample
from src.data.patch_dataset.sampling_strategies import random_sample
from src.data.patch_dataset.patch_descriptor import PatchDescriptor
from src.data.patch_dataset.utils import get_patch_size
from src.data.patch_dataset.utils import get_transformation

class PatchDataset(data.Dataset):
    """Dataset that returns patches from each image using a sampling strategy.

    Args:
        kitti_dataset:
            A KITTIDataset object which specifies the data to sample patches from, i.e. images, cameras and depth maps.
        aspect_ratio:
            Aspect ratio of sampled patches
        sampling_strategy:
            a string specifying which sampling strategy to apply. Possible choices are 'random', 'grid', 'corner'
        max_num_samples_per_image:
            Maximum number of patches to sample from one image
        preserve_camera:
            Whether to apply a warp that makes sure that the sampled patch was (virtually) acquired by a certain camera.
    """
    def __init__(
        self,
        kitti_dataset: KITTIDataset,
        base_size: tuple[int, int],

        scale: int,
        num_scales: int,
        scaling_factor: float,
        
        out_size: tuple[int, int],
        sampling_strategy: str,
        max_num_samples_per_image: int,
        preserve_camera: bool,
        return_descriptors: bool = False,
    ):
        self.kitti_dataset = kitti_dataset
        self.base_size = base_size
        self.out_size = out_size
        
        self.scale = scale
        self.num_scales = num_scales
        self.scaling_factor = scaling_factor

        self.max_num_samples_per_image = max_num_samples_per_image
        self.preserve_camera = preserve_camera
        self.return_descriptors = return_descriptors

        self.sampling_strategy: Callable[[int, np.ndarray, Camera, tuple[int, int], int, int, float], np.ndarray]
        match sampling_strategy:
            case 'random':
                self.sampling_strategy = random_sample
            case 'grid':
                self.sampling_strategy = grid_sample
            case 'corner':
                self.sampling_strategy = corner_sample
            case _:
                raise ValueError(f"Unrecognized sampling strategy \"{sampling_strategy}\"")
    
    def __len__(self):
        return len(self.kitti_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[PatchDescriptor]]:
        image: np.ndarray
        camera: Camera
        depth: np.ndarray
        image, camera, gt_depth, velo_depth = self.kitti_dataset[index]

        # Choose patches
        patch_descriptors: list[PatchDescriptor] = self.sample_patches(index, image, camera)
        
        patch_image_list: list[np.ndarray] = []
        gt_patch_depth_list: list[np.ndarray] = []
        velo_patch_depth_list: list[np.ndarray] = []

        gt_point_cloud: np.ndarray = depth2cloud(gt_depth, camera)
        #gt_point_cloud = np.hstack([gt_point_cloud, np.ones((gt_point_cloud.shape[0], 1), dtype=np.float32)])
        
        velo_point_cloud: np.ndarray = depth2cloud(velo_depth, camera)
        #velo_point_cloud = np.hstack([velo_point_cloud, np.ones((velo_point_cloud.shape[0], 1), dtype=np.float32)])

        # Extract the chosen patches
        for pd in patch_descriptors:
            pd.extract_from_image(image, gt_point_cloud, velo_point_cloud, gt_depth, velo_depth, self.preserve_camera)
            
            if self.kitti_dataset.mode == 'val' or self.kitti_dataset.mode == 'test':
                patch_image_list.append(pd.patch_image)
                gt_patch_depth_list.append(pd.patch_gt_depth)
                velo_patch_depth_list.append(pd.patch_velo_depth)
            else:
                hflip = (random.random() > 0.5)
                if hflip:
                    patch_image_list.append(pd.patch_image[:, ::-1])
                    gt_patch_depth_list.append(pd.patch_gt_depth[:, ::-1])
                    velo_patch_depth_list.append(pd.patch_velo_depth[:, ::-1])
                else:
                    patch_image_list.append(pd.patch_image)
                    gt_patch_depth_list.append(pd.patch_gt_depth)
                    velo_patch_depth_list.append(pd.patch_velo_depth)

        # To torch
        batch_image = np.stack(patch_image_list, axis=0).astype(np.float32) # shape = (B, H, W, 3)
        batch_gt_depth = np.stack(gt_patch_depth_list, axis=0).astype(np.float32) # shape = (B, H, W)
        batch_velo_depth = np.stack(velo_patch_depth_list, axis=0).astype(np.float32) # shape = (B, H, W)

        if self.return_descriptors:
            return torch.from_numpy(batch_image).permute(0, 3, 1, 2), torch.from_numpy(batch_gt_depth).unsqueeze(1), torch.from_numpy(batch_velo_depth).unsqueeze(1), patch_descriptors
        else:
            return torch.from_numpy(batch_image).permute(0, 3, 1, 2), torch.from_numpy(batch_gt_depth).unsqueeze(1), torch.from_numpy(batch_velo_depth).unsqueeze(1)
    
    def sample_patches(self, seed: int, image: np.ndarray, camera: Camera) -> list[PatchDescriptor]:
        
        # Have some reproducibility control for validation/test
        if self.kitti_dataset.mode == 'val' or self.kitti_dataset.mode == 'test':
            XY = self.sampling_strategy(seed, image, camera, self.base_size, self.scale, self.num_scales, self.scaling_factor)
            rng = np.random.default_rng(seed)
        else:
            XY = self.sampling_strategy(-1, image, camera, self.base_size, self.scale, self.num_scales, self.scaling_factor)
            rng = np.random.default_rng()
        rng.shuffle(XY, axis=0)
        H, W, _ = image.shape

        XY = np.vstack([XY, np.array([[W // 2, H // 2]], dtype=np.int64)]) # Safe point
        
        num_valid = 0
        patch_descriptors: list[PatchDescriptor] = []
        
        for center_x, center_y in XY:
            original_size = get_patch_size(self.base_size, self.scale, self.scaling_factor)
            tform, patch_camera, scaling_factor = get_transformation(center_x, center_y, self.scale, original_size, camera, self.out_size)

            pd = PatchDescriptor(
                center_x=center_x,
                center_y=center_y,
                scale=self.scale,
                original_size=original_size,
                actual_size=self.out_size,
                scaling_factor=scaling_factor,
                camera=patch_camera,
                tform=tform,
            )

            # Check if it has out-of-view pixels
            if pd.is_valid(image):
                patch_descriptors.append(pd)
                num_valid += 1

            if num_valid >= self.max_num_samples_per_image:
                break

        return patch_descriptors

def collate_fn(batch):
    """A custom collate function for using PatchDataset in a DataLoader as it returns already batched items."""
    patch_batches: list[torch.Tensor] = [patches for patches, _, _ in batch]
    gt_depth_batches: list[torch.Tensor] = [gt_depth for _, gt_depth, _ in batch]
    velo_depth_batches: list[torch.Tensor] = [velo_depth for _, _, velo_depth in batch]

    return torch.cat(patch_batches), torch.cat(gt_depth_batches), torch.cat(velo_depth_batches)