import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from src.data.kitti.dataset import KITTIDataset
from src.data.image_dataset.val_dataset import ValDataset
from src.data.image_dataset.train_dataset import TrainDataset
from src.logger import Logger
from src.loss import ScaleInvariantLoss
from src.loss import ScaleShiftInvariantLoss
from src.model.DepthNet import DepthNet
from src.setup import setup_and_get_startup_args
from src.trainer import Trainer
from src.train_cycle import train_cycle
from src.validator import Validator
from src.data.patch_dataset.utils import get_patch_size
from src.data.patch_dataset.utils import NUM_SCALES

parser = argparse.ArgumentParser()

# General options
parser.add_argument('--name', type=str, help="Experiment name")
parser.add_argument('--mode', type=str, default='train', help="Which subsets of the whole dataset to use, options are \"train\" or \"full\"")
parser.add_argument('--trainfrac', type=float, default=None, help="Percentage of training data to use")
parser.add_argument('--valfrac', type=float, default=None, help="Percentage of validation data to use")
parser.add_argument('--epochs', type=int, default=10, help="Number of epochs of training")
parser.add_argument('--batch', type=int, default=16, help="Batch size for training")
parser.add_argument('--workers', type=int, default=4, help="Number of workers for the training dataloader")
parser.add_argument('--modeldepth', type=int, default=50, help="ResNet depth: 18, 34, 50, 101, 152")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument("--stepsize", type=int, default=1, help="After how many epochs the lr is reduced")
parser.add_argument('--gamma', type=float, default=1., help="Multiplicative factor for the learning rate")

# Patch dataset options
parser.add_argument('--affine', action='store_true', help="Work with affine depth")
parser.add_argument('--half', action='store_true', help="Whether to half the size of the image before feeding the model.")

# Always put these, not really using them right now
parser.add_argument('--velo', action='store_true', help='Whether to use raw data or corrected projection')
parser.add_argument('--project', action='store_true', help='Whether to project raw data on image plane')

args = parser.parse_args()

def main():
    global args
    experiment_name = args.name
    print(f"EXPERIMENT: {experiment_name}")

    assert torch.cuda.is_available()
    device = 'cuda'

    # MODEL
    model: nn.Module = DepthNet(depth=args.modeldepth, input_size=(172, 576) if args.half else (344, 1152), share_encoder_for_confidence_prediction=True)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}\n")

    # OPTIMIZER
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-4},
        {'params': model.confidence_head.parameters(), 'lr': 1e-4},
    ], lr=args.lr)

    # SCHEDULER
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # LOSS
    criterion: nn.Module
    if args.affine:
        criterion = ScaleShiftInvariantLoss()
    else:
        criterion = ScaleInvariantLoss()

    # INTERNAL PARAMETERS
    internal_args: dict = setup_and_get_startup_args(
        experiment_name,
        model,
        optimizer,
        scheduler,
        args.epochs,
    )

    # Save command line options
    print(vars(args))
    path = os.path.join(internal_args['experiment_dir'], 'options.json')
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # TRAINER
    skip_valid = False
    if args.mode == 'full':
        skip_valid = True
        args.trainfrac = None
    
    train_kitti = KITTIDataset(args.mode, percentage=args.trainfrac, center_crop=False, from_velodyne=args.velo, project=args.project)
    train_set= TrainDataset(train_kitti, half_size=args.half)
    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    train_logger = Logger(internal_args['info_dir'], periodic_plot=True, period=50)
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        0., # Do not train the confidence head
        train_loader,
        args.affine,
        train_logger,
        internal_args['checkpoints_dir'],
    )

    # VALIDATOR
    val_kitti = KITTIDataset('val', percentage=args.valfrac, center_crop=True, from_velodyne=args.velo, project=True)
    val_set = ValDataset(val_kitti, args.half)
    val_loader = data.DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    val_logger = Logger(internal_args['info_dir'])    
    validator = Validator(model, val_loader, args.affine, val_logger)

    # Start train
    train_cycle(trainer, validator, internal_args['start_epoch'], args.epochs, skip_valid)

if __name__ == "__main__":
    # Warnings to be ignored:
    #     - Numpy casting from float to int warning int src.data.numpy_datasets.utils.py, function cloud2depth
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()