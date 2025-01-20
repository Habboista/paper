import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

from src.data.kitti.dataset import KITTIDataset
from src.data.patch_dataset.dataset import collate_fn
from src.data.patch_dataset.dataset import PatchDataset
from src.logger import Logger
from src.loss import ScaleInvariantLoss
from src.loss import ScaleShiftInvariantLoss
from src.model.DepthNet import DepthNet
from src.setup import setup_and_get_startup_args
from src.trainer import Trainer
from src.train_cycle import train_cycle
from src.validator import Validator
from src.data.patch_dataset.utils import get_patch_size

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
parser.add_argument('--strategy', type=str, default='random', help="Sampling strategy to use for extracting patches")

parser.add_argument('--baseH', type=int, default=100,  help="Height of base patch size")
parser.add_argument('--baseW', type=int, default=100,  help="Width of base patch size")

parser.add_argument('--outH', type=int, default=300,  help="Height of model input")
parser.add_argument('--outW', type=int, default=300,  help="Width of model input")

parser.add_argument('--wconf', type=float, default=0.01, help="Weight of confidence loss during training")
parser.add_argument('--shareencoder', action='store_true', help="Whether to share the encoder for confidence prediction with depth estimation model")

parser.add_argument('--preservecamera', action='store_true', help="Whether to perform a correction to the geometry of the patch or not")
parser.add_argument('--maxsamplesperimage', type=int, default=8, help="Maximum number of patches to be extracted from an image")

parser.add_argument('--scale', type=int, default=1, help="Which scale to work with")
parser.add_argument('--numscales', type=int, default=3, help="Number of scales to work with")
parser.add_argument('--scaling', type=float, default=1.5, help="Scaling factor")

args = parser.parse_args()

def main():
    global args
    experiment_name = args.name
    print(f"EXPERIMENT: {experiment_name}")

    assert torch.cuda.is_available()
    device = 'cuda'

    base_size = (args.baseH, args.baseW)
    out_size = (args.outH, args.outW)

    # MODEL
    model: nn.Module = DepthNet(depth=args.modeldepth, input_size=out_size, share_encoder_for_confidence_prediction=args.shareencoder)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}\n")

    # OPTIMIZER
    if args.shareencoder:
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-5},
            {'params': model.decoder.parameters(), 'lr': 1e-4},
            {'params': model.confidence_head.parameters(), 'lr': 1e-4},
        ], lr=args.lr)
    else:
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-5},
            {'params': model.decoder.parameters(), 'lr': 1e-4},
            {'params': model.confidence_head.parameters(), 'lr': 1e-4},
            {'params': model.confidence_encoder.parameters(), 'lr': 1e-5}
        ], lr=args.lr)

    # SCHEDULER
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    # LOSS
    if args.affine:
        criterion: nn.Module = ScaleShiftInvariantLoss()
    else:
        criterion: nn.Module = ScaleInvariantLoss()

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

    path = os.path.join(internal_args['experiment_dir'], 'description.txt')
    with open(path, 'a') as f:
        f.write("")

    # TRAINER
    skip_valid = False
    if args.mode == 'full':
        skip_valid = True
        args.trainfrac = None
    
    train_kitti = KITTIDataset(args.mode, percentage=args.trainfrac, center_crop=False)
    train_set = PatchDataset(
        train_kitti,
        base_size,
        args.scale,
        args.numscales,
        args.scaling,
        model.input_size,
        args.strategy,
        args.maxsamplesperimage,
        args.preservecamera,
        False,
    )
    train_loader = data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    train_logger = Logger(internal_args['info_dir'], periodic_plot=True, period=50)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        conf_weight=args.wconf,
        gt_weight=1.,
        velo_weight=0.,
        reg_weight=0.,
        data_loader=train_loader,
        affine=args.affine,
        logger=train_logger,
        checkpoints_dir=internal_args['checkpoints_dir'],
    )

    # VALIDATOR
    val_kitti = KITTIDataset('val', percentage=args.valfrac, center_crop=False)
    val_set = PatchDataset(
        val_kitti,
        base_size,
        args.scale,
        args.numscales,
        args.scaling,
        model.input_size,
        args.strategy,
        args.maxsamplesperimage,
        args.preservecamera,
        False,
    )
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
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