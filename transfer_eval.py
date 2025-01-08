import os
import torch

from src.data.kitti.dataset import KITTIDataset
from src.data.patch_dataset.dataset import PatchDataset
from src.data.patch_dataset.dataset import collate_fn
from src.validator import Validator
from src.model.DepthNet import DepthNet
from src.logger import Logger

kitti = KITTIDataset('test')
strategies = ['random', 'grid', 'corner']
model = DepthNet(depth=50, input_size=(100, 100), share_encoder_for_confidence_prediction=True, pretrained=True)
model.load_state_dict(torch.load(os.path.join('src', 'results', 'random_warp', 'run_01', 'checkpoints', 'model_checkpoint_010.pth')))
model = model.to('cuda')
for strat in strategies:
    print(strat.upper())
    patch_dataset = PatchDataset(
        kitti,
        base_size=(100, 100),
        scale=2,
        num_scales=3,
        scaling_factor=1.5,
        out_size=(100, 100),
        sampling_strategy=strat,
        max_num_samples_per_image=4,
        preserve_camera=True,
        return_descriptors=False,
    )
    validator = Validator(
        model,
        torch.utils.data.DataLoader(patch_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn),
        False,
        Logger(os.path.join('src', 'test_results')),
        kitti.min_depth,
        kitti.max_depth,
    )
    validator.eval()