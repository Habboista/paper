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
parser.add_argument('--experiment', type=str, default='proposed_crop_40', help="Which experiment model to evaluate")
args = parser.parse_args()

model = DepthNet(depth=50, input_size=(300, 300), share_encoder_for_confidence_prediction=True, pretrained=True)
model.load_state_dict(torch.load(os.path.join('src', 'results', 'coarse_corner_crop', 'run_01', 'checkpoints', 'model_checkpoint_040.pth'), weights_only=True, map_location='cpu'))
model = model.to('cuda')

def main():
    kitti = KITTIDataset('test', center_crop=False, from_velodyne=False)
    full_image_eval_set = ValDataset(kitti, False)
    patch_eval_set = PatchDataset(
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
    eval_logger = Logger(eval_dir)
    eval_logger.new_epoch()

    loop_bar = tqdm(range(len(kitti)), desc='EVALUATING', colour='yellow')
    for i in loop_bar:
        image_patches, _, _, descriptors = patch_eval_set[i]

        image_patches = image_patches.to('cuda')

        model.eval()
        with torch.no_grad():
            batch_pred, conf_scores = model(image_patches)

        batch_pred = batch_pred.to('cpu')
        conf_scores = conf_scores.to('cpu')
        
        image, camera, gt_depth, _ = kitti[i]
        
        pred_depth: np.ndarray = merge(image, camera, image_patches, batch_pred, descriptors, conf_scores)
        
        info = compute_metrics(np.log(pred_depth), gt_depth, 1e-3, 80., False)
        eval_logger.log_info(info)
        loop_bar.set_postfix({'a1': info['a1']})
    
    eval_logger.print_last_epoch_summary()
    eval_logger.save_last_epoch_summary(args.experiment)

if __name__ == "__main__":
    # Warnings to be ignored:
    #     - Numpy casting from float to int warning int src.data.numpy_datasets.utils.py, function cloud2depth
    #     - Torch masked module that is in beta
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()