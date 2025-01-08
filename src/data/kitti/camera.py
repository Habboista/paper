from typing import Optional

import numpy as np
import skimage.transform as T

_H: int = 256
_W: int = 256

class Camera:
    def __init__(
        self,
        K: np.ndarray = np.eye(3, 3, dtype=np.float32),
        R: np.ndarray = np.eye(3, 3, dtype=np.float32),
        t: np.ndarray = np.zeros((3, 1), dtype=np.float32),
        shape: tuple[int, int] = (_H, _W),
    ):
        self.K = K
        self.R = R
        self.t = t
        self.shape = shape

    def get_signature(self) -> tuple[int, int, int, int, int, int]:
        H = self.shape[0]
        W = self.shape[1]
        fx = int(self.K[0, 0])
        fy = int(self.K[1, 1])
        px = int(self.K[0, 2])
        py = int(self.K[1, 2])
        return H, W, fx, fy, px, py

    def get_projective_intrinsics(self) -> np.ndarray:
        return np.hstack([self.K, np.zeros((3, 1), dtype=np.float32)])
    
    def get_projective_extrinsics(self) -> np.ndarray:
        return np.vstack([
            np.hstack([self.R, self.t]),
            np.array([[0, 0, 0, 1]], dtype=np.float32),
        ])

    def copy(self):
        return Camera(self.K.copy(), self.R.copy(), self.t.copy(), self.shape)
    
    def cam2img(self, x_cam: np.ndarray) -> np.ndarray:
        x_img: np.ndarray = np.zeros((2,), dtype=np.float32)
        x_img[0] = self.K[0, 0] * (x_cam[0] / x_cam[2]) + self.K[0, 2]
        x_img[1] = self.K[1, 1] * (x_cam[1] / x_cam[2]) + self.K[1, 2]
        return x_img
    
    def img2cam(self, x_img: np.ndarray, z: Optional[float] = None) -> np.ndarray:
        x_cam: np.ndarray = np.ones((3,), dtype=np.float32)
        x_cam[0] = (x_img[0] - self.K[0, 2]) / self.K[0, 0]
        x_cam[1] = (x_img[1] - self.K[1, 2]) / self.K[1, 1]

        if z is not None:
            x_cam *= z

        return x_cam
    
    def cam2world(self, x_cam: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def world2cam(self, x_world: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def world2img(self, x_world: np.ndarray) -> np.ndarray:
        return self.cam2img(self.world2cam(x_world))
    
    def img2world(self, x_img: np.ndarray) -> np.ndarray:
        return self.cam2world(self.img2cam(x_img))

    def center_crop(self, crop_size: tuple[int, int]):
        out_cam = self.copy()
        out_cam.K[0, 2] = crop_size[1] / 2
        out_cam.K[1, 2] = crop_size[0] / 2
        out_cam.shape = crop_size
        return out_cam
    
    def rotate(self, rot_mat: np.ndarray):
        out_cam = self.copy()
        out_cam.R = rot_mat @ out_cam.R
        out_cam.t = rot_mat @ out_cam.t
        return out_cam

    def scale_through_intrinsics(self, s: float):
        raise NotImplementedError
    
    def scale_through_extrinsics(self, s: float):
        raise NotImplementedError
