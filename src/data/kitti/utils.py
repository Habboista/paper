import os

import numpy as np
import torch

from src.data.kitti.camera import Camera

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def readlines(filename: str) -> list[str]:
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def get_camera_parameters(calib_dir: str, cam: int) -> Camera:
    """Get camera parameters from KITTI calibration file.

    Args:
        calib_dir: path to KITTI calibration file
        cam: camera number, 2 = left, 3 = right

    Returns:
        dictionary with camera parameters
        {
            "K": np.ndarray of shape (3, 3),
            "R": np.ndarray of shape (3, 3),
            "t": np.ndarray of shape (3, 1),
            "image_size": np.ndarray of shape (2,) corresponding to image height and width
        }
    """
    # There are various reference systems:
    #   - velodyne reference system,
    #   - camera reference system both for left and right camera,
    #   - rectified camera reference both for left and right camera
    
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # transformation velodyne->camera
    R_t_velo2cam: np.ndarray = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    R_t_velo2cam = np.vstack((R_t_velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    S: np.ndarray = cam2cam[f"S_rect_0{cam}"][::-1]

    # transformation camera->rectified camera
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

    # projection matrix
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)

    # Decompose camera matrix
    K: np.ndarray = P_rect[:, :3]
    R = np.eye(3)
    p4 = P_rect[:, 3]

    t = np.array([
        [p4[0] / K[0, 0]],
        [0.             ],
        [0.             ],
    ])

    R_t_rect2rect = np.vstack((
            np.hstack((R,        t)),
            np.array([0, 0, 0, 1.0])
        ))
    
    # transformation velodyne -> rectified camera
    R_t = R_t_rect2rect @ R_cam2rect @ R_t_velo2cam

    R = R_t[:3, :3] # (3, 3) shaped
    t = R_t[:3, 3:] # (3, 1) shaped

    camera: Camera = Camera(
        K.astype(np.float32),
        R.astype(np.float32),
        t.astype(np.float32),
        tuple(S.astype(np.int32)),
    )

    return camera

def get_velo_points(filename: str) -> np.ndarray:
    """Load velodyne points from filename and remove all points located behind the cameras image planes. """

    # each row of the velodyne data is (forward, left, up, reflectance)
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4) # N x 4

    # 4th coordinate is the reflectance, ignore it
    points[:, 3] = 1.0  # homogeneous coords

    # Consider only points in front of the cameras
    points = points[points[:, 0] > 0.27, :] # 0.27 is the distance between the lidar and the camera

    return points

def depth2cloud(depth: np.ndarray, camera: Camera, k: int = 1) -> np.ndarray:
    """Project depth map to 3D point cloud.
    
    Args:
        depth_map:
            Numpy array of shape H x W and dtype float32
        camera:
            camera object
        k:
            How fine is the grid, it can be thought as the number of points to be sampled between pixels.
            If k=1 (default), then one point per pixel is returned,
        
    Returns:
        Numpy array of shape N x 3 and dtype float32
    """
    x: np.ndarray = np.linspace(0., float(depth.shape[1] - 1), depth.shape[1] * k, dtype=np.float32) + 0.5
    y: np.ndarray = np.linspace(0., float(depth.shape[0] - 1), depth.shape[0] * k, dtype=np.float32) + 0.5

    grid_x, grid_y = np.meshgrid(x, y, indexing='xy')

    x = grid_x.flatten()
    y = grid_y.flatten()
    
    z: np.ndarray = depth[y.astype(np.int64), x.astype(np.int64)]

    valid = (z > 0)
    x = x[valid]
    y = y[valid]
    z = z[valid]

    x = (x - camera.K[0, 2]) / camera.K[0, 0]
    y = (y - camera.K[1, 2]) / camera.K[1, 1]

    x *= z
    y *= z

    x_cam = np.vstack([x, y, z]) # shape = (3, HW)

    x_world = np.linalg.inv(camera.R) @ (x_cam - camera.t) # shape = (3, HW)
    
    return x_world.T # shape = (HW, 3)

def cloud2depth(
    X_world: np.ndarray, camera: Camera
) -> np.ndarray:
    """Project points to the image plane and render their depth map.

    Args:
        X_world:
            point cloud of shape N x 4: (x, y, z, 1) w.r.t. to the reference system specified by extrinsic camera_parameters

        camera:
            Camera object describing the camera w.r.t. which render the depth map

    Returns:
        depth map of shape (H, W), of dtype float32. Positive values are valid depth values, zero values are invalid pixels (for which no point is projected onto).
    """

    ## project the points to camera reference system
    #R_t = camera.get_projective_extrinsics()
    #
    #X_cam = X_world @ R_t.T
    #X_cam = X_cam[:, :3] / X_cam[:, 3:] # dehomogenize
    
    # Ignore projective coord
    X_world = X_world[:, :3] #
    X_cam = X_world @ camera.R.T + camera.t.T
    
    # Consider only points in front of the camera
    valid = (X_cam[:, 2] > 0)
    X_cam = X_cam[valid]

    # project points to image
    X_img = (X_cam @ camera.K.T)
    X_img[:, :2] = X_img[:, :2] / X_img[:, 2:]
    
    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    x = (np.round(X_img[:, 0]).astype(np.int32) - 1)
    y = (np.round(X_img[:, 1]).astype(np.int32) - 1)
    z = X_cam[:, 2]
    valid_inds = (x >= 0) & (y >= 0) & (x < camera.shape[1]) & (y < camera.shape[0])
    
    x = x[valid_inds]
    y = y[valid_inds]
    z = z[valid_inds]

    # draw depth map
    depth: np.ndarray = np.zeros(tuple(camera.shape), dtype=np.float32) # shape (H, W)
    depth[y, x] = z

    # find the duplicate points and choose the closest depth
    flat_inds = np.ravel_multi_index((y, x), tuple(camera.shape))
    duplicate_inds, counts = np.unique(flat_inds, return_counts=True)
    duplicate_inds = duplicate_inds[counts > 1]
    for dd in duplicate_inds:
        pts = np.nonzero(flat_inds == dd)[0]
        depth[y[pts], x[pts]] = z[pts].min()

    return depth

def image_np2torch(np_image: np.ndarray, normalize: bool = False) -> torch.Tensor:
    initial_type = np_image.dtype
    np_image = np_image.astype(np.float32)
    if initial_type == np.uint8:
        np_image /= 255.
    
    if normalize:
        np_image = (np_image - MEAN) / STD
    
    t_image: torch.Tensor = torch.from_numpy(np_image).permute(2, 0, 1)
    return t_image

def image_torch2np(t_image: torch.Tensor, unnormalize: bool = False) -> np.ndarray:
    np_image: np.ndarray = t_image.squeeze().cpu().permute(1, 2, 0).numpy()
    if unnormalize:
        np_image = STD * np_image + MEAN
    return np_image

def depth_np2torch(np_depth: np.ndarray) -> torch.Tensor:
    np_depth = np_depth.astype(np.float32)
    t_depth: torch.Tensor = torch.from_numpy(np_depth).unsqueeze(0)
    return t_depth

def depth_torch2np(t_depth: torch.Tensor) -> np.ndarray:
    np_depth: np.ndarray = t_depth.squeeze().detach().cpu().numpy()
    return np_depth