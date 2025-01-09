from dataclasses import dataclass
from typing import Optional

import numpy as np
import skimage
from skimage.transform import ProjectiveTransform as Transform

from src.data.kitti.camera import Camera
from src.data.kitti.utils import cloud2depth
from src.data.kitti.utils import depth2cloud

@dataclass
class PatchDescriptor:
    center_x: int
    center_y: int
    scale: int

    original_size: tuple[int, int]
    actual_size: tuple[int, int]
    scaling_factor: float

    camera: Camera
    tform: Transform

    patch_depth: Optional[np.ndarray] = None
    preserve_camera: Optional[bool] = None

    def is_valid(self, image: np.ndarray) -> bool:
        H, W, _ = image.shape
        coords = self.get_original_coords()
        if (coords[:, 0] < 0).any() or (coords[:, 0] >= W).any() or (coords[:, 1] < 0).any() or (coords[:, 1] >= H).any():
            return False
        return True
    
    def get_original_coords(self) -> np.ndarray:
        size = self.actual_size
        
        coords = np.array([
            [0.,           0.],
            [size[1],      0.],
            [size[1], size[0]],
            [0.,      size[0]],
        ])
        coords = self.tform.inverse(coords)
        return coords
    
    def project_to_image(self, image_camera: Camera) -> np.ndarray:
        if self.patch_depth is None:
            raise ValueError("No depth associated")
        else:
            if self.preserve_camera:
                # Get initial image coords
                x = np.linspace(0., float(self.patch_depth.shape[1] - 1), self.patch_depth.shape[1], dtype=np.int64)
                y = np.linspace(0., float(self.patch_depth.shape[0] - 1), self.patch_depth.shape[0], dtype=np.int64)

                grid_x, grid_y = np.meshgrid(x, y, indexing='xy')

                y = grid_y.flatten()
                x = grid_x.flatten()
                
                z = self.patch_depth[y, x]

                valid = (z > 0)
                x_img = x[valid]
                y_img = y[valid]

                cloud: np.ndarray = depth2cloud(self.patch_depth, self.camera)
                
                # roto-translate point cloud
                xyz_cam_rototranslated = cloud @ image_camera.R.T + image_camera.t.T

                temp_patch_depth = self.patch_depth.copy()
                temp_patch_depth[y_img, x_img] = xyz_cam_rototranslated[:, 2]

                image_depth = skimage.transform.warp(temp_patch_depth, self.tform, output_shape=image_camera.shape, order=0)#, clip=False, preserve_range=True) # dtype=np.float64
                
                #cloud = np.hstack([cloud, np.ones((cloud.shape[0], 1), dtype=np.float32)])
                #image_depth: np.ndarray = cloud2depth(cloud, image_camera)
                return image_depth
            else:
                image_depth: np.ndarray = np.zeros(image_camera.shape, dtype=np.float32)
                h, w = self.original_size
 
                patch_depth = skimage.transform.resize(self.patch_depth, output_shape=self.original_size, order=0, anti_aliasing=False)

                h_top = h // 2
                h_bottom = h - h_top
                w_left = w // 2
                w_right = w - w_left

                image_depth[
                    self.center_y - h_top : self.center_y + h_bottom,
                    self.center_x - w_left   : self.center_x + w_right,
                ] = patch_depth

                return image_depth

    def extract_from_image(
        self,
        image: np.ndarray,
        gt_point_cloud: np.ndarray,
        velo_point_cloud: np.ndarray,
        gt_depth: np.ndarray,
        velo_depth: np.ndarray,
        preserve_camera: bool,
    ) -> None:
        
        patch_image: np.ndarray
        patch_gt_depth: np.ndarray
        patch_velo_depth: np.ndarray

        if preserve_camera:
            # Warp
            patch_image = skimage.transform.warp(image, self.tform.inverse, output_shape=self.actual_size, mode='edge') # range [0., 1.], dtype=np.float64
            
            # GT

            # Get initial image coords
            x = np.linspace(0., float(gt_depth.shape[1] - 1), gt_depth.shape[1], dtype=np.int64)
            y = np.linspace(0., float(gt_depth.shape[0] - 1), gt_depth.shape[0], dtype=np.int64)

            grid_x, grid_y = np.meshgrid(x, y, indexing='xy')

            y = grid_y.flatten()
            x = grid_x.flatten()
            
            z = gt_depth[y, x]

            valid = (z > 0)
            x_img = x[valid]
            y_img = y[valid]

            # roto-translate point cloud
            xyz_cam_rototranslated = gt_point_cloud @ self.camera.R.T + self.camera.t.T

            temp_gt_depth = gt_depth.copy()
            temp_gt_depth[y_img, x_img] = xyz_cam_rototranslated[:, 2]

            patch_gt_depth = skimage.transform.warp(temp_gt_depth, self.tform.inverse, output_shape=self.actual_size, order=0)#, clip=False, preserve_range=True) # dtype=np.float64
            
            # VELO

            # Get initial image coords
            x = 0.5 + np.linspace(0., float(velo_depth.shape[1] - 1), velo_depth.shape[1], dtype=np.float32)
            y = 0.5 + np.linspace(0., float(velo_depth.shape[0] - 1), velo_depth.shape[0], dtype=np.float32)

            grid_x, grid_y = np.meshgrid(x, y, indexing='xy')

            x = grid_x.flatten().astype(np.int64)
            y = grid_y.flatten().astype(np.int64)
            
            z = velo_depth[y, x]

            valid = (z > 0)
            x_img = x[valid]
            y_img = y[valid]

            # roto-translate point cloud
            xyz_cam_rototranslated = velo_point_cloud @ self.camera.R.T + self.camera.t.T

            temp_velo_depth = velo_depth.copy()
            temp_velo_depth[y_img, x_img] = xyz_cam_rototranslated[:, 2]

            patch_velo_depth = skimage.transform.warp(temp_velo_depth, self.tform.inverse, output_shape=self.actual_size, order=0)#, clip=False, preserve_range=True) # dtype=np.float64
            
            #patch_gt_depth: np.ndarray = cloud2depth(gt_point_cloud, self.camera) # dtype=np.float32
            #patch_velo_depth: np.ndarray = cloud2depth(velo_point_cloud, self.camera) # dtype=np.float32
        else:
            # Crop
            h, w = self.original_size
            h_top = h // 2
            h_bottom = h - h_top
            w_left = w // 2
            w_right = w - w_left

            patch_image = image[
                self.center_y - h_top : self.center_y + h_bottom,
                self.center_x - w_left   : self.center_x + w_right,
                :,
            ]
            patch_gt_depth = gt_depth[
                self.center_y - h_top : self.center_y + h_bottom,
                self.center_x - w_left   : self.center_x + w_right,
            ]
            patch_velo_depth = velo_depth[
                self.center_y - h_top : self.center_y + h_bottom,
                self.center_x - w_left   : self.center_x + w_right,
            ]
            patch_image = skimage.transform.resize(patch_image, output_shape=self.actual_size)
            patch_gt_depth = skimage.transform.resize(patch_gt_depth, output_shape=self.actual_size, order=0, anti_aliasing=False)
            patch_velo_depth = skimage.transform.resize(patch_velo_depth, output_shape=self.actual_size, order=0, anti_aliasing=False)
        
        self.patch_image = patch_image
        self.patch_gt_depth = patch_gt_depth
        self.patch_velo_depth = patch_velo_depth
        self.preserve_camera = preserve_camera