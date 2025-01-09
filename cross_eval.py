import argparse
import os
from typing import Optional

import numpy as np
import skimage
import torch
from tqdm import tqdm

from src.logger import Logger
from src.data.kitti.dataset import KITTIDataset
from src.data.patch_dataset.dataset import PatchDataset
from src.data.image_dataset.val_dataset import ValDataset
from src.inference import merge
from src.model.DepthNet import DepthNet
from src.validator import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='corrected', help="Which experiment model to evaluate")
args = parser.parse_args()

model_warp = DepthNet(depth=50, input_size=(300, 300), share_encoder_for_confidence_prediction=True, pretrained=True)
model_warp.load_state_dict(torch.load(os.path.join('src', 'results', 'coarse_corner_warp', 'run_01', 'checkpoints', 'model_checkpoint_040.pth'), weights_only=True, map_location='cpu'))
model_warp = model_warp.to('cuda')

model_crop = DepthNet(depth=50, input_size=(300, 300), share_encoder_for_confidence_prediction=True, pretrained=True)
model_crop.load_state_dict(torch.load(os.path.join('src', 'results', 'coarse_corner_crop', 'run_01', 'checkpoints', 'model_checkpoint_040.pth'), weights_only=True, map_location='cpu'))
model_crop = model_crop.to('cuda')

baseline = DepthNet(depth=50, input_size=(344, 1152), share_encoder_for_confidence_prediction=True, pretrained=True)
baseline.load_state_dict(torch.load('src/results/baseline/run_01/checkpoints/model_checkpoint_040.pth', weights_only=True, map_location='cpu'))
baseline = baseline.to('cuda')

baseline_half = DepthNet(depth=50, input_size=(172, 576), share_encoder_for_confidence_prediction=True, pretrained=True)
baseline_half.load_state_dict(torch.load('src/results/baseline_half/run_01/checkpoints/model_checkpoint_040.pth', weights_only=True, map_location='cpu'))
baseline_half = baseline_half.to('cuda')

def main():
    kitti = KITTIDataset('test', center_crop=False, from_velodyne=False)

    full_image_eval_set = ValDataset(kitti, False)
    half_image_eval_set = ValDataset(kitti, True)
    patch_eval_set_warp = PatchDataset(
        kitti_dataset=kitti,
        base_size=(100, 100),
        scale=2,
        num_scales=3,
        scaling_factor=1.5,
        out_size=(300, 300),
        sampling_strategy='grid',
        max_num_samples_per_image=16,
        preserve_camera=True, ################### WARP
        return_descriptors=True,
    )
    patch_eval_set_crop = PatchDataset(
        kitti_dataset=kitti,
        base_size=(100, 100),
        scale=2,
        num_scales=3,
        scaling_factor=1.5,
        out_size=(300, 300),
        sampling_strategy='grid',
        max_num_samples_per_image=16,
        preserve_camera=False, ################### CROP
        return_descriptors=True,
    )

    eval_dir = os.path.join('src', 'test_results')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    eval_logger_proposed_warp = Logger(eval_dir)
    eval_logger_proposed_warp.new_epoch()

    eval_logger_proposed_crop = Logger(eval_dir)
    eval_logger_proposed_crop.new_epoch()

    eval_logger_baseline_full = Logger(eval_dir)
    eval_logger_baseline_full.new_epoch()

    eval_logger_baseline_half = Logger(eval_dir)
    eval_logger_baseline_half.new_epoch()

    loop_bar = tqdm(range(len(kitti)), desc='EVALUATING', colour='yellow')
    for i in loop_bar:
        image_b, _, _ = full_image_eval_set[i]
        image_h, _, _ = half_image_eval_set[i]
        image_patches_warp, _, _, descriptors_warp = patch_eval_set_warp[i]
        image_patches_crop, _, _, descriptors_crop = patch_eval_set_crop[i]

        image_patches_warp = image_patches_warp.to('cuda')
        image_patches_crop = image_patches_crop.to('cuda')
        image_b = image_b.to('cuda')
        image_h = image_h.to('cuda')

        model_warp.eval()
        model_crop.eval()
        baseline.eval()
        baseline_half.eval()

        with torch.no_grad():
            batch_pred_warp, conf_scores = model_warp(image_patches_warp)

            batch_pred_crop, conf_scores = model_crop(image_patches_crop)

            pred_b, _ = baseline(image_b[None])
            pred_b = torch.exp(torch.clamp(pred_b, np.log(1e-3), np.log(80.)))[0]

            pred_h, _ = baseline_half(image_h[None])
            pred_h = torch.exp(torch.clamp(pred_h, np.log(1e-3), np.log(80.)))[0]

        pred_b = pred_b.to('cpu')
        pred_h = pred_h.to('cpu')
        batch_pred_warp = batch_pred_warp.to('cpu')
        batch_pred_crop = batch_pred_crop.to('cpu')
        conf_scores = conf_scores.to('cpu')
        
        image, camera, gt_depth, _ = kitti[i]
        
        pred_depth_warp: np.ndarray = merge(image, camera, image_patches_warp, batch_pred_warp, descriptors_warp, conf_scores)
        pred_depth_crop: np.ndarray = merge(image, camera, image_patches_crop, batch_pred_crop, descriptors_crop, conf_scores)
        pred_b = pred_b.squeeze().cpu().numpy()
        pred_h = pred_h.squeeze().cpu().numpy()

        if pred_depth_warp.shape != pred_h.shape:
            pred_h = skimage.transform.resize(pred_h, pred_depth_warp.shape, order=0, anti_aliasing=False)

        if pred_depth_warp.shape != pred_b.shape:
            pred_b = skimage.transform.resize(pred_b, pred_depth_warp.shape, order=0, anti_aliasing=False)

        assert pred_depth_warp.shape == pred_depth_crop.shape
        assert pred_depth_crop.shape == pred_b.shape
        assert pred_b.shape == pred_h.shape

        valid = (pred_depth_warp > 0) & (pred_b > 0) & (pred_depth_crop > 0) & (pred_h > 0)
        
        pred_depth_warp[~valid] = 0.
        pred_depth_crop[~valid] = 0.
        pred_b[~valid] = 0.
        pred_h[~valid] = 0.

        info_proposed_warp = compute_metrics(np.log(pred_depth_warp), gt_depth, 1e-3, 80., False)
        info_proposed_crop = compute_metrics(np.log(pred_depth_crop), gt_depth, 1e-3, 80., False)
        info_baseline = compute_metrics(np.log(pred_b), gt_depth, 1e-3, 80., False)
        info_baseline_half = compute_metrics(np.log(pred_h), gt_depth, 1e-3, 80., False)

        eval_logger_proposed_warp.log_info(info_proposed_warp)
        eval_logger_proposed_crop.log_info(info_proposed_crop)
        eval_logger_baseline_full.log_info(info_baseline)
        eval_logger_baseline_half.log_info(info_baseline_half)
        # loop_bar.set_postfix({'a1': info['a1']})
    
    print("PROPOSED WARP")
    eval_logger_proposed_warp.print_last_epoch_summary()
    eval_logger_proposed_warp.save_last_epoch_summary(args.experiment + "_proposed_warp")

    print("PROPOSED CROP")
    eval_logger_proposed_crop.print_last_epoch_summary()
    eval_logger_proposed_crop.save_last_epoch_summary(args.experiment + "_proposed_crop")
    
    print("BASSLINE FULL")
    eval_logger_baseline_full.print_last_epoch_summary()
    eval_logger_baseline_full.save_last_epoch_summary(args.experiment + "_baseline_full")

    print("BASSLINE HALF")
    eval_logger_baseline_half.print_last_epoch_summary()
    eval_logger_baseline_half.save_last_epoch_summary(args.experiment + "_baseline_half")

if __name__ == "__main__":
    # Warnings to be ignored:
    #     - Numpy casting from float to int warning int src.data.numpy_datasets.utils.py, function cloud2depth
    #     - Torch masked module that is in beta
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()