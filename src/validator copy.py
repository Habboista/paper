from dataclasses import dataclass
from typing import Callable
from typing import Self

import numpy as np
import skimage.transform as T
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from src.logger import Logger
from src.data.kitti.dataset import KITTIDataset
from src.data.kitti.utils import depth_torch2np
from src.data.patch_dataset.dataset import PatchDataset

@dataclass
class Validator:
    model: nn.Module
    valid_set: KITTIDataset
    affine: bool
    logger: Logger
    min_depth: float = 1e-3
    max_depth: float = 80
    base_size: tuple[int, int] = (300, 300)
    scale: int = 0
    num_scales: int = 1
    scaling_factor: float = 1.
    out_size: tuple[int, int] = (300, 300)
    sampling_strategy: str = 'grid'
    preserve_camera: bool = True
    
    def eval(self, checkpoint_name=None):
        patch_eval_set = PatchDataset(
            kitti_dataset=self.valid_set,
            base_size=self.base_size,
            scale=self.scale,
            num_scales=self.num_scales,
            scaling_factor=self.scaling_factor,
            out_size=self.out_size,
            sampling_strategy=self.sampling_strategy,
            max_num_samples_per_image=float('inf'),
            preserve_camera=self.preserve_camera,
            return_descriptors=True,
        )

        loop_bar = tqdm(range(len(self.valid_set)), desc='EVALUATING', colour='blue')
        self.logger.new_epoch()
        
        # Loop the validation set
        for i in loop_bar:
            image_patches, _, _, descriptors = patch_eval_set[i]

            image_patches = image_patches.to('cuda') # shape = (B, 3, H_image, W_image)

            # batch predict
            self.model.eval()
            with torch.no_grad():
                batch_pred, conf_scores = self.model(image_patches) # shape = (B, 1, H_pred, W_pred)

            image, camera, gt_depth, _ = self.valid_set[i]
            batch_pred = batch_pred.to('cpu')
            conf_scores = conf_scores.to('cpu')
            pred_depth: np.ndarray = merge(image, camera, image_patches, batch_pred, descriptors, conf_scores)

            # unbatch
            for b in range(B):

                # convert to numpy
                np_pred: np.ndarray =  depth_torch2np(batch_pred[b]) # shape = (H_pred, W_pred)
                np_gt_depth: np.ndarray =  depth_torch2np(batch_gt_depth[b]) # shape = (H_depth, W_depth)
                np_velo_depth: np.ndarray =  depth_torch2np(batch_velo_depth[b]) # shape = (H_depth, W_depth)
                
                pred_conf: float = conf_scores[b].item()

                if (np_velo_depth > 0).any():

                    # compute depth metrics
                    velo_depth_metrics = compute_metrics(
                        np_pred,
                        np_velo_depth,
                        self.min_depth,
                        self.max_depth,
                        self.affine
                    )
                    self.logger.log_info({'velo_' + k: v for k, v in velo_depth_metrics.items()})

                # are there enough depth values?
                if (np_gt_depth > 0).any():

                    # compute depth metrics
                    gt_depth_metrics = compute_metrics(
                        np_pred,
                        np_gt_depth,
                        self.min_depth,
                        self.max_depth,
                        self.affine
                    )
                    self.logger.log_info({'gt_' + k: v for k, v in gt_depth_metrics.items()})
                    self.logger.log_info({'confL1': abs(pred_conf - gt_depth_metrics['a1'])})
                else:
                    # the patch is out of distribution
                    self.logger.log_info({'confL1': abs(pred_conf - 0.)})
        
        # Display statistics
        self.logger.print_last_epoch_summary()
        if checkpoint_name is not None:
            self.logger.save_last_epoch_summary("validation_metrics_" + checkpoint_name)
    
def compute_metrics(pred: np.ndarray, depth: np.ndarray, min_depth: float, max_depth: float, affine: bool) -> dict[str, float]:
    pred # shape = (H_pred, W_pred)
    depth # shape = (H_depth, W_depth)
        
    # Resize pred to match gt size
    if pred.shape != depth.shape:
        print("resizing for evaluation")
        pred = T.resize(pred, depth.shape, order=0, anti_aliasing=False)
    
    # Consider only valid pixels
    valid = (depth > 0) & (~np.isnan(pred)) & (pred > 0)
    if not valid.any():
        raise ValueError("No valid pixels in the ground-truth depth map")
    
    depth = depth[valid] # shape = (N,)
    pred = pred[valid] # shape = (N,)

    if not affine:
        pred_depth = np.exp(pred)
    else:
        pred_disp = pred
        pred_disp = align_scale_and_shift(pred_disp, 1 / depth)

        valid = (pred_disp > 0)
        depth = depth[valid] # shape = (M,)
        pred_disp = pred_disp[valid] # shape = (M,)

        pred_depth = 1 / pred_disp

    # Clamp prediction and ground truth
    depth = depth.clip(min_depth, max_depth)
    pred_depth = pred_depth.clip(min_depth, max_depth)

    # Compute accuracy metrics
    thresh = np.max(np.stack([(depth / pred_depth), (pred_depth / depth)], axis=0), axis=0)
    mask_a1 = (thresh < 1.25)
    mask_a2 = (thresh < 1.25 ** 2)
    mask_a3 = (thresh < 1.25 ** 3)
    a1 = (mask_a1.mean(dtype=np.float32).item() if mask_a1.any() else 0.)
    a2 = (mask_a2.mean(dtype=np.float32).item() if mask_a2.any() else 0.)
    a3 = (mask_a3.mean(dtype=np.float32).item() if mask_a3.any() else 0.)

    # Compute error metrics
    rmse = (depth - pred_depth) ** 2
    rmse = np.sqrt(rmse.mean()).item()

    rmse_log = (np.log(depth) - np.log(pred_depth)) ** 2
    rmse_log = np.sqrt(rmse_log.mean()).item()

    abs_rel = np.mean(np.abs(depth - pred_depth) / depth).item()

    sq_rel = np.mean(((depth - pred_depth) ** 2) / depth).item()

    d = np.log(depth) - np.log(pred_depth)
    si_err = (np.mean(d**2) - np.mean(d)**2).item()

    return {
        #'val_percentage': val_percentage,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'si_err': si_err,
    }

def align_scale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x - x.mean() + np.log(y).mean()

def align_scale_and_shift(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns the first tensor aligned to the second."""
    
    x = x[:, None]
    ones = np.ones_like(x)
    M = np.hstack([x, ones])
    s, t = np.linalg.lstsq(M, y)[0]
    x = s * x[:, 0] + t

    return x

def _align_scale_and_shift(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    x_vec = np.stack((x, np.ones_like(x)), axis=1) # (M, 2)
    x_mat = x_vec[..., None] # (M, 2, 1)

    A = np.matmul(x_mat, np.swapaxes(x_mat, 1, 2)).sum(0) # (2, 2)
    try:
        A = np.linalg.inv(A) 
    except np.linalg.LinAlgError:
        print("Warning: singular matrix, using pseudo inverse")
        A = np.linalg.pinv(A)

    b = (x_vec * y[..., None]).sum(0) # (2,)
    s, t = A @ b
    x = s*x + t
    print("min", x.min(),"max", x.max())
    exit()
    return x