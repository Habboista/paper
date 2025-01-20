import os

import numpy as np
import skimage

from src.data.kitti.camera import Camera
from src.data.patch_dataset.patch_descriptor import PatchDescriptor

PATCH_AREA_SCALE_0 = 300*300
NUM_SCALES = 1
PATCH_SIZE_AT_SCALE: list[tuple[int, int]] = []
DOWNSCALE_FACTOR: float = 1.1

def get_patch_size(base_size: tuple[int, int], scale: int, scaling_factor: float) -> tuple[int, int]:
    h_prev, w_prev = base_size
    for s in range(0, scale):
        h_next = int(h_prev * scaling_factor)
        w_next = int(w_prev * scaling_factor)
        h_prev, w_prev = h_next, w_next

    return h_prev, w_prev

def get_depth_scaling_matrix(s: float, projective: bool = False) -> np.ndarray:
    matrix: np.ndarray = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0.,  s],
    ], dtype=np.float32)

    if projective:
        matrix = np.hstack([matrix, np.zeros((3, 1), dtype=np.float32)])
        bottom_row = np.array([0., 0., 0., 1.], dtype=np.float32)
        matrix = np.vstack([matrix, bottom_row])

    return matrix

def get_rotation_matrix(x: float, y: float, cam: Camera, projective: bool = False, inverse: bool = False) -> np.ndarray:
    """Compute the rotation that the camera has to do for making the pixel at (x, y) the principal point.
    
    Args:
        x:
            x-coordinate of the pixel (column)
        y:
            y-coordinate of the pixel (row)
        camera_parameters:
            dict of arrays describing the camera

    Returns:
        The 4x4 rotation matrix R (as projective transform).
        Precisely: R * [0, 0, f] projects onto the image plane in [x, y].
        Normally R * [0, 0, f] projects onto the camera principal point.
    """

    # Compute angles of rotation
    theta_zx: float = np.arctan2(x - cam.K[0, 2], cam.K[0, 0])
    theta_zy: float = np.arctan2(y - cam.K[1, 2], cam.K[1, 1])
    
    if inverse:
        theta_zx *= -1
        theta_zy *= -1

    # Rotation matrices
    R_xz = np.array([
        [ np.cos(theta_zx), 0,    np.sin(theta_zx)],
        [                0, 1,                   0],
        [-np.sin(theta_zx), 0,    np.cos(theta_zx)],
    ], dtype=np.float32)

    R_yz = np.array([
        [1,                    0,                   0],
        [0,     np.cos(theta_zy),    np.sin(theta_zy)],
        [0,    -np.sin(theta_zy),    np.cos(theta_zy)],
    ], dtype=np.float32)
    
    R: np.ndarray = R_yz @ R_xz
    if projective:
        R = np.hstack([R, np.zeros((3, 1), dtype=np.float32)])
        bottom_row = np.array([0., 0., 0., 1.], dtype=np.float32)
        R = np.vstack([R, bottom_row])
    return R

def get_transformation(x: int, y: int, s: int, size: tuple[int, int], camera: Camera, out_size: tuple[int, int]):
    rot_mat: np.ndarray = get_rotation_matrix(float(x), float(y), camera, inverse=True)
    rot_cam = camera.rotate(rot_mat)

    crop_size: tuple[int, int] = size
    scaling_factor: float = crop_size[0] / out_size[0]
    
    scal_mat: np.ndarray = get_depth_scaling_matrix(scaling_factor)
    scal_cam = rot_cam.copy()
    # Zoom out by moving the image plane closer to the camera
    # (We also fix the starting focal length to 720.)
    scal_cam.K[0, 0] = 720. / scaling_factor
    scal_cam.K[1, 1] = 720. / scaling_factor

    crop_camera = scal_cam.center_crop(out_size)

    H = crop_camera.K @ rot_mat @ np.linalg.pinv(camera.K)

    return skimage.transform.ProjectiveTransform(H), crop_camera, scaling_factor

def get_transformation_bak(x: int, y: int, s: int, size: tuple[int, int], camera: Camera, out_size: tuple[int, int]):
    rot_mat: np.ndarray = get_rotation_matrix(float(x), float(y), camera, inverse=True)
    rot_cam = camera.rotate(rot_mat)

    crop_size: tuple[int, int] = size
    scaling_factor: float = crop_size[0] / out_size[0]

    scal_mat: np.ndarray = get_depth_scaling_matrix(scaling_factor)
    scal_cam = rot_cam.rotate(scal_mat)

    crop_camera = scal_cam.center_crop(out_size)
    crop_camera.K[0, 0] = 720.
    crop_camera.K[1, 1] = 720.

    H = crop_camera.K @ scal_mat @ rot_mat @ np.linalg.pinv(camera.K)

    return skimage.transform.ProjectiveTransform(H), crop_camera, scaling_factor
    
def detect_points_of_interest(
    image: np.ndarray,
    edge_map: np.ndarray,
    min_distance: int,
    use_corners: bool,
    valid_mask: np.ndarray,
    #borders_specifiers: tuple[float, float, float, float],
) -> np.ndarray:
    """Returns the coordinates of detected corners in the image.
    
    The result is a tensor of shape N x 2 corresponding to rows and columns of the image.
    Borders are excluded from the corner detection. Default border specifiers are appropriate for KITTI images.

    Args:
        image: image as a numpy array of shape H x W x 3
        min_distance: minimal distance between two detected corners
        borders_specifiers: (i1, i2, j1, j2) - top, bottom, left, right borders in [0, 1] range
    Returns:
        detected corners as a numpy array of shape N x 2, encoded as (row, col)
    """
    if use_corners:
        # Compute corner response
        gray_image = skimage.color.rgb2gray(image)
        corner_response = skimage.feature.corner_moravec(gray_image)
        # corner_response = skimage.feature.corner_moravec(corner_response)
        
        # Consider only edges
        corner_response[edge_map < 1e-2] = 0
        interest_map = corner_response
    else:
        interest_map = edge_map

    # Exclude image borders
    # h, w, _ = image.shape
    # i1 = int(borders_specifiers[0] * h)
    # i2 = int(borders_specifiers[1] * h)
    # j1 = int(borders_specifiers[2] * w)
    # j2 = int(borders_specifiers[3] * w) 
    # interest_map[:i1, :] = 0
    # interest_map[i2:, :] = 0
    # interest_map[:, :j1] = 0
    # interest_map[:, j2:] = 0
    interest_map[~valid_mask] = 0

    # Sample peaks
    peaks: np.ndarray = skimage.feature.peak_local_max(
        interest_map,
        min_distance=min_distance,
    ) # N x 2, encoded as (row, col)

    if peaks.shape[0] == 0:
        raise ValueError('No corners detected')
    
    return peaks

EDGE_DETECTOR_DEVICE = 'cuda'

def generate_grid(image: np.ndarray, size: tuple[int, int], scale: int, relative_x_overlap: float = 0.75, relative_y_overlap: float = 0.75):
    H, W, _ = image.shape
    h, w = size
    
    delta_x = int(w * (1 - relative_x_overlap))
    delta_y = int(h * (1 - relative_y_overlap))

    N_x = (W - w) // delta_x
    N_y = (H - w) // delta_y

    # I want the center always to be considered
    # since it's the only valid point of the grid independent of the patch size
    if N_x % 2 == 0:
        N_x += 1
    if N_y % 2 == 0:
        N_y += 1

    grid_x, grid_y = np.meshgrid(
        np.concatenate([
            np.linspace(w//2, W//2, (N_x - 1)//2 + 1, endpoint=True, dtype=np.int64)[:-1],
            np.linspace(W//2, W - (w - w//2), (N_x - 1)//2 + 1, endpoint=True, dtype=np.int64),
        ]),
        np.concatenate([
            np.linspace(h//2, H//2, (N_y - 1)//2 + 1, endpoint=True, dtype=np.int64)[:-1],
            np.linspace(H//2, H - (h - h//2), (N_y - 1)//2 + 1, endpoint=True, dtype=np.int64),
        ]),
        indexing='xy'
    )

    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    #grid_s = scale * np.ones_like(grid_x)
    return grid_x, grid_y #, grid_s

def generate_initial_patches(image: np.ndarray, aspect_ratio: float):
    patch_descriptors: list[dict] = []

    size = get_patch_size(aspect_ratio, 0)
    for x, y, s in zip(*generate_grid(image, aspect_ratio, 0)):
        pd = {
            'center': (x, y),
            'size': size,
            'scale': s,
        }
    
    patch_descriptors.append(pd)
    return patch_descriptors

    # Extract a simple grid patch.
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [j - blsize, k - blsize, j - blsize + 2 * blsize, k - blsize + 2 * blsize]
            patch_bound = [box[0] + patchbounds[1], box[1] + patchbounds[0], patchbounds[3] - patchbounds[1],
                           patchbounds[2] - patchbounds[0]]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list

def get_rect(x, y, h, w):
    
    start = [(y - h // 2, x - w // 2)]
    end = [(y + h // 2, x + w // 2)]
    return start, end

def copy_descriptor(pd: dict):
    return {k: pd[k] for k in pd.keys()}


def is_inbound(rect, H, W) -> bool:
    if rect[0][0][0] < 0 or rect[0][0][1] < 0 or rect[1][0][0] < 0 or rect[1][0][1] < 0 or \
            rect[0][0][0] >= H or rect[0][0][1] >= W or rect[1][0][0] >= H or rect[1][0][1] >= W:
        return False
    return True

# Adaptively select patches
def adaptiveselection(integral_grad, XYS, aspect_ratio, optimal_edge_density) -> np.ndarray:
    keep: np.ndarray = np.ones((XYS.shape[0],), dtype=np.bool_)
    count = 0
    H, W = integral_grad.shape

    # Go through all patches
    for i, (x, y, s) in enumerate(XYS):

        # Get patch
        h, w = get_patch_size(aspect_ratio, s)
        rect = get_rect(x, y, h, w)

        # Check if in-bound
        if not is_inbound(rect, H, W):
            keep[i] = 0
            continue 

        # Compute the amount of gradients present in the patch from the integral image.
        patch_edge_density = (skimage.transform.integrate(integral_grad, *rect) / (h * w))[0]
        
        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if patch_edge_density >= optimal_edge_density:

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:
                
                if s + 1 >= NUM_SCALES:
                    break

                s += 1

                h, w = get_patch_size(aspect_ratio, s)
                rect = get_rect(x, y, h, w)

                # If we not within the image, stop
                if not is_inbound(rect, H, W):
                    s -= 1
                    break

                # If edge density is smaller, stop
                patch_edge_density = (skimage.transform.integrate(integral_grad, *rect) / (h * w))[0]
                if patch_edge_density < optimal_edge_density:
                    break
        
            XYS[i, 2] = s
        
        # Not enough edges, drop it
        else:

            keep[i] = 0
    
    # Return selected patches
    return XYS[keep]