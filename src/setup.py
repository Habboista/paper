from typing import Any

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

def setup_and_get_startup_args(
    experiment_name: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    num_epochs: int
) -> dict[str, Any]:

    # Directories
    experiment_dir = os.path.join(os.path.dirname(__file__), 'results', experiment_name)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    
    # Find number of runs
    start_epoch: int
    start_new_run: bool = False
    run_list = os.listdir(experiment_dir)
    run_list = [d for d in run_list if os.path.isdir(os.path.join(experiment_dir, d))] # exclude options file
    if len(run_list) == 0:
        start_new_run = True
        start_epoch: int = 1

    else:
        last_run: int = len(run_list)
        last_run_dir = os.path.join(experiment_dir, 'run_' + str(last_run).zfill(2))

        # Checkpoints and info directories
        last_run_checkpoints_dir = os.path.join(last_run_dir, 'checkpoints')
        last_run_info_dir = os.path.join(last_run_dir, 'info')
        if not os.path.isdir(last_run_checkpoints_dir):
            os.makedirs(last_run_checkpoints_dir)
        if not os.path.isdir(last_run_info_dir):
            os.makedirs(last_run_info_dir)

        # Load checkpoints
        print("Looking for a checkpoint...")
        checkpoints = os.listdir(last_run_checkpoints_dir)
        checkpoints.sort() # Makes sure that checkpoints are in alphabetical order!

        if len(checkpoints) == 0:
            start_epoch = 1
        else:
            start_epoch: int = int(checkpoints[-1].split('_')[-1].split('.')[0]) + 1

        if start_epoch > num_epochs:
            # There are as many checkpoints as epochs, start a new run
            print(f"Found run-{last_run:02} completed.\n")
            start_new_run = True
            start_epoch = 1
        else:
            print(f"Continuing run-{last_run:02}.\n")

            # Is there a checkpoint to start from?
            if len(checkpoints) > 0:

                print(f"Checkpoint {checkpoints[-1]} found, loading model, optimizer and scheduler.\n" \
                    f"Resuming from epoch: {start_epoch}\n")
                
                model_checkpoint_path = os.path.join(last_run_checkpoints_dir, checkpoints[-3])
                model.load_state_dict(torch.load(model_checkpoint_path))

                optimizer_checkpoint_path = os.path.join(last_run_checkpoints_dir, checkpoints[-2])
                optimizer.load_state_dict(torch.load(optimizer_checkpoint_path))

                scheduler_checkpoint_path = os.path.join(last_run_checkpoints_dir, checkpoints[-1])
                scheduler.load_state_dict(torch.load(scheduler_checkpoint_path))

            else:
                print("No checkpoint found, instantiating new model, optimizer and scheduler.\n")
    
    if start_new_run:
        num_runs: int = len(run_list)
        run = num_runs + 1
        print(f"Starting run-{run:02}.\n")
        checkpoints_dir = os.path.join(experiment_dir, 'run_' + str(run).zfill(2), 'checkpoints')
        info_dir = os.path.join(experiment_dir, 'run_' + str(run).zfill(2), 'info')
        os.makedirs(checkpoints_dir)
        os.makedirs(info_dir)
    else:
        checkpoints_dir = last_run_checkpoints_dir
        info_dir = last_run_info_dir
    
    return {
        'experiment_dir': experiment_dir,
        'start_epoch': start_epoch,
        'checkpoints_dir': checkpoints_dir,
        'info_dir': info_dir,
    }