import torch
import torch.nn as nn
from torch import Tensor

class ScaleShiftInvariantLoss(nn.Module):
    def __init__(self, trim: bool = False):
        super().__init__()
        self.trim = trim

    def align(self, d: Tensor) -> Tensor:
        t: Tensor = torch.median(d, dim=-1, keepdim=True)[0]
        s: Tensor = torch.mean(torch.abs(d - t), dim=-1, keepdim=True)

        if s.item() > 0.:
            return (d - t) / s
        else:
            return d - t

    def forward(self, batched_pred: Tensor, batched_gt: Tensor) -> Tensor:
        """Forward pass.

        Args:
            batched_pred: a tensor of shape (B, 1, H, W)
            batched_gt: a tensor shaped like batched_pred but its values must be >= 0
        
        Returns:
            Scale and shift invariant loss, 0. if batched_gt is all zero
        """
        # batched_pred shape = (B, 1, H, W)
        # batched_gt shape = (B, 1, H, W)

        B = batched_pred.shape[0]
        loss: Tensor = torch.tensor(0., requires_grad=True)
        
        batched_pred = batched_pred.view(B, -1) # shape = (B, H * W)
        batched_gt = batched_gt.view(B, -1) # shape = (B, H * W)
            
        for pred, gt in zip(batched_pred, batched_gt):
            
            valid = (gt > 0)
            if not valid.any():
                B -= 1
                continue

            gt = gt[valid]
            pred = pred[valid]

            # Pass to disparity space
            gt = 1. / gt

            # Range [0, 1]
            gt = (gt - gt.min()) / (gt.max() - gt.min())

            # Align
            pred = self.align(pred)
            gt = self.align(gt)

            # Compute error
            error = torch.abs(pred - gt)
            M = len(error)
            if self.trim:
                error, _ = torch.sort(error, dim=0, descending=True)
                error = error[int(0.2 * M):]

            loss = loss + torch.sum(error) / (2. * M)

        if B > 0:
            loss = loss / B
        else:
            print("Warning: no valid predictions")

        return loss
    
class ScaleInvariantLoss(nn.Module):
    """Compute the scale invariant loss used by Eigen et al.
    
    Raises:
        ValueError
            if the ground truth has no valid pixels (i.e. is all zeros)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, depth: Tensor) -> Tensor:
        pred # shape = (B, 1, H_pred, W_pred)
        depth # shape = (B, 1, H_depth, W_depth)

        # Resize prediction
        #if pred.shape != depth.shape:
        #    pred = nn.functional.interpolate(pred, size=depth.shape[-2:], mode="nearest") # shape = (B, 1, H_depth, W_depth)

        # Flatten
        B = pred.shape[0]
        pred = pred.view(B, -1) # shape = (B, H_depth * W_depth)
        depth = depth.view(B, -1) # shape = (B, H_depth * W_depth)

        # Exclude invalid pixels
        valid = (depth > 0) # shape = (B, H_depth * W_depth)

        # Compute loss
        loss = torch.tensor(0., requires_grad=True)
        num_valid_samples = B
        
        for pred, gt, val in zip(pred, depth, valid):

            if not val.any():
                num_valid_samples -= 1
                continue

            d = (torch.log(gt) - pred)[val] # shape = (V,)
            sample_loss = torch.mean(d**2) - 0.5 * torch.mean(d)**2
            loss = loss + sample_loss

        if num_valid_samples == 0:
            raise ValueError("No valid pixels to train on")
        
        return loss / num_valid_samples