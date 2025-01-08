import numpy as np
import torch
import skimage

from src.data.kitti.camera import Camera
from src.data.patch_dataset.sampling_strategies import grid_sample
from src.data.patch_dataset.patch_descriptor import PatchDescriptor
from src.model.DepthNet import DepthNet

def merge(
    image: torch.Tensor,
    camera: Camera,
    batch_image_patches: torch.Tensor,
    batch_pred_patches: torch.Tensor,
    patch_descriptors: list[PatchDescriptor],
    confidence_scores: torch.Tensor
) -> np.ndarray:
    
    np_batch_pred: np.ndarray = batch_pred_patches.cpu().squeeze(1).numpy() # shape = (B, H_patch, W_patch)

    image_depth_list: list[np.ndarray] = []
    weights = []
    h, w = patch_descriptors[0].actual_size
    x = np.linspace(0., float(w-1), w, dtype=np.float32)
    y = np.linspace(0., float(h-1), h, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')
    base_weight = np.exp(-((grid_x - w/2)**2 + (grid_y - h/2)**2) / (2*w))
    
    for pred, pd, conf in zip(np_batch_pred, patch_descriptors, confidence_scores):
        pred = np.clip(pred, a_min=None, a_max=np.log(200.))
        pd.patch_depth = np.exp(pred)

        weight_iamge = skimage.transform.warp(base_weight, inverse_map=pd.tform, output_shape=image.shape[:-1])
        weights.append(weight_iamge)
        # if conf.item() > 0.9:
        image_depth_list.append(pd.project_to_image(camera))
    
    pred = np.stack(image_depth_list, axis=0)
    weights = np.stack(weights, axis=0)
    valid = (pred > 0)
    
    # Align
    #t_pred = torch.tensor(pred)
    #m = (t_pred > 0)
#
    #s = torch.zeros(t_pred.shape[0], 1, 1, requires_grad=True)
    #t = torch.zeros(t_pred.shape[0], 1, 1, requires_grad=True)
    #print("START HERE", t_pred.min(), t_pred.max())
    #for _ in range(10):
    #    x = torch.exp(s) * t_pred + t
    #    mu = (m * x).sum(0, keepdim=True) / (m.sum(0, keepdim=True) + 1e-9) # shape (1, H, W)
    #    print(mu.min(), mu.max())
    #    loss = (((x - mu)**2).sum(0) / (m.sum(0) + 1e-9)**2).mean() + (t + s)**2 / torch.tensor(res)[:, None, None]
    #    print(loss)
    #    loss.backward()
    #    with torch.no_grad():
    #        s -= 1e-5 * s.grad
    #        t -= 1e-5 * t.grad
    #        s.grad = None
    #        t.grad = None
    #
    #t_pred = torch.exp(s) * t_pred + t
    #t_pred[m] = 0.
    #pred = t_pred.numpy()

    # Aggregate
    # Mean
    pred[~valid] = np.nan
    weights[~valid] = np.nan

    #pred = np.log(pred)
    pred = np.nansum(pred * weights, axis=0) / np.nansum(weights, axis=0)
    #pred = np.exp(pred)

    # Max
    #res = np.max(pred, axis=0)
    #nan = np.isnan(res)
    #res[nan] = 0.
    return pred