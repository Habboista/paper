from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from tqdm import tqdm

from src.data.kitti.utils import depth_torch2np
from src.logger import Logger
from src.validator import compute_metrics

@dataclass
class Trainer:
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: lr_scheduler.LRScheduler
    criterion: nn.Module
    conf_weight: float
    data_loader: data.DataLoader
    affine: bool
    logger: Logger
    checkpoints_dir: str
    
    def check_parameters(self):
        for p in self.model.parameters():
            if p.data.isnan().any():
                raise ValueError("NaN detected in model parameters")
            if p.data.isinf().any():
                raise ValueError("Inf detected in model parameters")
    
    def check_gradients(self):
        for p in self.model.parameters():
            if p.grad.isnan().any():
                raise ValueError("NaN detected in gradients")
            if p.grad.isinf().any():
                raise ValueError("Inf detected in gradients")
            
    def train(self, *, checkpoint_name: Optional[str] = None) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.logger.new_epoch()
        print(f"Learning rate = {self.scheduler.get_last_lr()}")

        loop_bar = tqdm(self.data_loader, desc='TRAINING', colour='green')
        for image, gt_depth, velo_depth in loop_bar:

            # convert to device
            image = image.to('cuda') # shape = (B, 3, H_image, W_image)
            gt_depth = gt_depth.to('cuda') # shape = (B, 1, H_depth, W_depth)
            velo_depth = velo_depth.to('cuda') # shape = (B, 1, H_depth, W_depth)

            # predict
            pred, pred_conf_scores = self.model(image) # shape = (B, 1, H_pred, W_pred)

            # Identify patches with ground truth depth values
            gt_valid_samples: list[int] = []
            gt_valid_mask: torch.Tensor = (gt_depth > 0)
            for idx, vm in enumerate(gt_valid_mask):
                if vm.any():
                    gt_valid_samples.append(idx)

            velo_valid_samples: list[int] = []
            velo_valid_mask: torch.Tensor = (velo_depth > 0)
            for idx, vm in enumerate(velo_valid_mask):
                if vm.any():
                    velo_valid_samples.append(idx)

            # Initialize confidence scores
            conf_scores = torch.zeros(image.shape[0], 1).to('cuda')

            # Depth loss
            gt_depth_loss: torch.Tensor = torch.tensor(0.)
            velo_depth_loss: torch.Tensor = torch.tensor(0.)
            if len(gt_valid_samples) > 0:
                gt_depth_loss = self.criterion(pred[gt_valid_samples], gt_depth[gt_valid_samples])

                # Use only gt depth for confidence scores
                for idx in gt_valid_samples:
                    conf_scores[idx, 0] = compute_metrics(depth_torch2np(pred[idx]), depth_torch2np(gt_depth[idx]), 1e-3, 80., self.affine)['a1']
            
            if len(velo_valid_samples) > 0:
                velo_depth_loss = self.criterion(pred[velo_valid_samples], velo_depth[velo_valid_samples])

            # Confidence loss
            conf_loss: torch.Tensor = nn.functional.l1_loss(pred_conf_scores, conf_scores)

            # Regularization
            l1_reg = torch.tensor(0., requires_grad=True)
            num_params = 0
            for p in self.model.parameters():
                l1_reg = l1_reg + torch.sum(torch.abs(p))
                num_params += p.numel()
            l1_reg = l1_reg / num_params

            # Total loss
            loss = gt_depth_loss + 0.1 * velo_depth_loss + self.conf_weight * conf_loss + 0.01 * l1_reg

            # backpropagation
            loss.backward()
            #self.check_gradients()

            self.optimizer.step()
            #self.check_parameters()

            # Saving training info
            loop_bar.set_postfix({'Loss': loss.item(), 'std': torch.std(pred, dim=(1, 2, 3)).mean().item()})
            loss_info = dict()
            loss_info['loss'] = loss.detach().item()
            if len(gt_valid_samples) > 0:
                loss_info['gt_depth_loss'] = gt_depth_loss.detach().item()
            if len(velo_valid_samples) > 0:
                loss_info['velo_depth_loss'] = velo_depth_loss.detach().item()
            loss_info['conf_loss'] = conf_loss.detach().item()
            loss_info['l1_reg'] = l1_reg.detach().item()
            self.logger.log_info(loss_info)

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Free memory (sometimes it doesn't automatically free)
            del loss, image, gt_depth, velo_depth, pred
        
        # Update learning rate
        self.scheduler.step()

        # Save checkpoint
        if checkpoint_name is not None:

            # Remove old checkpoints for not saturating memory
            old_checkpoints = os.listdir(self.checkpoints_dir)
            for old_check in old_checkpoints:
                os.remove(os.path.join(self.checkpoints_dir, old_check))

            # Save new checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, 'model_' + checkpoint_name + '.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoints_dir, 'optimizer_' + checkpoint_name + '.pth'))
            torch.save(self.scheduler.state_dict(), os.path.join(self.checkpoints_dir, 'scheduler_' + checkpoint_name + '.pth'))

        # Display statistics
        self.logger.print_last_epoch_summary()
        if checkpoint_name is not None:
            self.logger.save_last_epoch_summary("training_loss_" + checkpoint_name)