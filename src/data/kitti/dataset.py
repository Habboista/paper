import os
from typing import Optional

import numpy as np
import skimage.io as io

from src.data.kitti.camera import Camera
from src.data.kitti.utils import cloud2depth
from src.data.kitti.utils import depth2cloud
from src.data.kitti.utils import get_camera_parameters
from src.data.kitti.utils import get_velo_points
from src.data.kitti.utils import readlines

PATH = os.path.expanduser('~/local_datasets/kitti')

class KITTIDataset:
    """Each item is a tuple of a uint8 numpy IMAGE, a CAMERA object, a float32 numpy DEPTH map.
    Pixels with no depth associated are given 0 depth.

    Args:
        mode:
            Which subset of the whole dataset to use, as defined by the Eigen split.
            'train', 'val', 'test' or 'full' = 'train' + 'val'
        percentage:
            Percentage of data to use from the subset specified by mode.
            No random, just take the first part. If None, use the whole subset specified by mode.
        center_crop:
            Whether to crop images around the camera principal point, in this case
            all images have the same shape = (344, 1152, 3).
            If False, images have different shapes.
        from_velodyne:
            Whether to use depth maps directly projected from raw Velodyne data (True) or
            use post-processed depth maps (False).
        project:
            if True, then project velodyne point cloud on image plane during runtime, otherwise the resulting depth map is loaded.
            If from_velodyne is False, this has no effect.
    """
    def __init__(self, mode: str, percentage: Optional[float] = None, center_crop: bool = False):
        self.mode = mode
        self.data_path: str = PATH
        self.filenames: list[str]
        self.center_crop = center_crop

        self.min_depth: float = 1e-3
        self.max_depth: float = 80.

        #if not from_velodyne or not project:
        #    prefix = "kitti_"
        #else:
        #    prefix = "eigen_"
        prefix = "kitti_"

        match mode:
            case "train":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", prefix + "train_files.txt")
                self.filenames = readlines(filenames_path)
            case "val":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", prefix + "val_files.txt")
                self.filenames = readlines(filenames_path)
            case "full":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", prefix + "train_files.txt")
                train_filenames = readlines(filenames_path)

                filenames_path = os.path.join(os.path.dirname(__file__), "split", prefix + "val_files.txt")
                val_filenames = readlines(filenames_path)

                self.filenames = train_filenames + val_filenames
            case "test":
                filenames_path = os.path.join(os.path.dirname(__file__), "split", prefix + "test_files.txt")
                self.filenames = readlines(filenames_path)
            case _:
                raise ValueError(f"Unknown mode {mode}")
            
        if percentage is not None:
            assert 0. < percentage and percentage < 1.
            N = len(self.filenames)
            self.filenames = self.filenames[:int(N * percentage)]

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, Camera, np.ndarray, np.ndarray]:
        image: np.ndarray
        camera: Camera
        depth: np.ndarray

        line = self.filenames[index].split(' ')

        if len(line) != 3:
            raise ValueError(f"line {index} does not contain 3 fields")
        folder, frame_index, side = line

        image = self.load_image(folder, frame_index, side)
        camera = self.load_camera_parameters(folder, side)

        velo_depth = self.load_velo_depth_map(folder, frame_index, side)
        gt_depth = self.load_gt_depth_map(folder, frame_index, side)

        if self.center_crop:
            px = camera.K[0, 2]
            py = camera.K[1, 2]
            camera = camera.center_crop((172*2, 576*2))
            image = image[
                int(py - 172):int(py + 172),
                int(px - 576):int(px + 576),
                :
            ]
            gt_depth = gt_depth[
                int(py - 172):int(py + 172),
                int(px - 576):int(px + 576),
            ]
            velo_depth = velo_depth[
                int(py - 172):int(py + 172),
                int(px - 576):int(px + 576),
            ]

        return image, camera, gt_depth, velo_depth

    def load_image(self, folder: str, frame_index: str, side: str) -> np.ndarray:
        """Load RGB image as a numpy array of shape (H, W, 3) and dtype=uint8."""
        fn = f"{int(frame_index):010d}.png"
        if side not in ["l", "r"]:
            raise ValueError(f"Unknown side {side}")
        filename = os.path.join(
            self.data_path,
            folder,
            "image_02" if side == "l" else "image_03",
            "data",
            fn,
        )
        image: np.ndarray = io.imread(filename) # (H, W, 3), uint8
        
        return image
    
    def load_point_cloud(self, folder: str, frame_index: str, side: str) -> np.ndarray:
        """Load velodyne point cloud as numpy array of shape (N, 4) and dtype=float32.
        The fourth coordinate is a 1.
        """
        filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points",
            "data",
            f"{int(frame_index):010d}.bin",
        )
        point_cloud: np.ndarray = get_velo_points(filename)
        return point_cloud
    
    def load_gt_depth_map(self, folder: str, frame_index: str, side: str) -> np.ndarray:
        """Load depth map as a numpy array of shape (H, W) and dtype=float32, values are expressed in meters."""
        folder = folder.split('/')[-1]
        filename_train_folder = os.path.join(
            self.data_path,
            'train',
            folder,
            "proj_depth",
            "groundtruth",
            "image_02" if side == "l" else "image_03",
            f"{int(frame_index):010d}.png",
        )
        filename_val_folder = os.path.join(
            self.data_path,
            'val',
            folder,
            "proj_depth",
            "groundtruth",
            "image_02" if side == "l" else "image_03",
            f"{int(frame_index):010d}.png",
        )
        try:
            depth_map: np.ndarray = io.imread(filename_train_folder, as_gray=True).astype(np.float32) / 256.0 # shape (H, W)
        except FileNotFoundError:
            depth_map: np.ndarray = io.imread(filename_val_folder, as_gray=True).astype(np.float32) / 256.0 # shape (H, W)

        return depth_map
    
    def load_velo_depth_map(self, folder: str, frame_index: str, side: str) -> np.ndarray:
        """Load depth map as a numpy array of shape (H, W) and dtype=float32, values are expressed in meters."""
        folder = folder.split('/')[-1]
        filename_train_folder = os.path.join(
            self.data_path,
            'train',
            folder,
            "proj_depth",
            "velodyne_raw",
            "image_02" if side == "l" else "image_03",
            f"{int(frame_index):010d}.png",
        )
        filename_val_folder = os.path.join(
            self.data_path,
            'val',
            folder,
            "proj_depth",
            "velodyne_raw",
            "image_02" if side == "l" else "image_03",
            f"{int(frame_index):010d}.png",
        )
        try:
            depth_map: np.ndarray = io.imread(filename_train_folder, as_gray=True).astype(np.float32) / 256.0 # shape (H, W)
        except FileNotFoundError:
            depth_map: np.ndarray = io.imread(filename_val_folder, as_gray=True).astype(np.float32) / 256.0 # shape (H, W)

        return depth_map
    
    def load_camera_parameters(self, folder: str, side: str) -> Camera:
        calib_path = os.path.join(self.data_path, folder.split("/")[0])
        camera: Camera = get_camera_parameters(calib_path, 2 if side == "l" else 3)
        return camera