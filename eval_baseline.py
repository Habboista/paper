import argparse
import os
from typing import Optional

import numpy as np
import skimage
import torch
from tqdm import tqdm

from src.logger import Logger
from src.data.kitti.dataset import KITTIDataset
from src.data.image_dataset.val_dataset import ValDataset
from src.inference import merge
from src.model.DepthNet import DepthNet
from src.validator import compute_metrics
from src.data.kitti.utils import depth_torch2np

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='baseline', help="Which experiment model to evaluate")
parser.add_argument('--epochs', type=int, default=40, help="Number of epochs that the model was trained")
parser.add_argument('--half', action='store_true', help="Whether to half the size of the input image")
parser.add_argument('--velo', action='store_true', help="Use velo data for evaluation")
args = parser.parse_args()

model = DepthNet(depth=50, input_size=(172, 576) if args.half else (344, 1152), share_encoder_for_confidence_prediction=True, pretrained=True)
model.load_state_dict(torch.load(os.path.join('src', 'results', args.experiment, 'run_01', 'checkpoints', f'model_checkpoint_{args.epochs:03d}.pth'), weights_only=True))

def main():
    kitti = KITTIDataset('test', center_crop=True)
    full_image_eval_set = ValDataset(kitti, args.half)

    eval_dir = os.path.join('src', 'test_results')
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    eval_logger = Logger(eval_dir)
    eval_logger.new_epoch()

    loop_bar = tqdm(range(len(kitti)), desc='EVALUATING', colour='yellow')
    for i in loop_bar:
        image, gt_depth, velo_depth = full_image_eval_set[i]

        model.eval()
        with torch.no_grad():
            pred, _ = model(image[None])
            pred = pred[0]

        pred = depth_torch2np(pred)
        gt = depth_torch2np(gt_depth)
        velo = depth_torch2np(velo_depth)
        if args.velo:
            info = compute_metrics(pred, velo, kitti.min_depth, kitti.max_depth, False)
        else:
            info = compute_metrics(pred, gt, kitti.min_depth, kitti.max_depth, False)
        
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