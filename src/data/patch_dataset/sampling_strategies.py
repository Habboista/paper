import numpy as np
import skimage
import skimage.transform as T

from src.data.kitti.camera import Camera
from src.data.patch_dataset.utils import get_transformation
from src.data.patch_dataset.utils import get_patch_size
from src.data.patch_dataset.utils import generate_grid
from src.data.patch_dataset.utils import adaptiveselection

# Generating local patches to perform the local refinement described in section 6 of the main paper.
def edge_sample(
    index: int,
    image: np.ndarray,
    camera: Camera,
    aspect_ratio: float,
) -> np.ndarray:
    
    # Compute the gradients as a proxy of the contextual cues.
    image_gray = skimage.color.rgb2gray(image)
    grad_map: np.ndarray = np.abs(skimage.filters.sobel_h(image_gray)) + np.abs(skimage.filters.sobel_v(image_gray))

    threshold = grad_map[grad_map > 0].mean()
    grad_map[grad_map < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    optimal_edge_density = grad_map.mean()
    grad_integral_map: np.ndarray = skimage.transform.integral_image(grad_map)

    # Get initial Grid
    grid_x, grid_y, grid_s = generate_grid(image, aspect_ratio, 0)
    XYS = np.stack([grid_x, grid_y, grid_s], axis=1)

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    XYS = adaptiveselection(grad_integral_map, XYS, aspect_ratio, optimal_edge_density)

    return XYS

def corner_sample(
    seed: int,
    image: np.ndarray,
    camera: Camera,
    base_size: tuple[int, int],
    scale: int,
    num_scales: int,
    scaling_factor: float,
) -> np.ndarray:
    
    gray = skimage.color.rgb2gray(image)
    H, W = gray.shape
    H = H // (scaling_factor**scale)
    W = W // (scaling_factor**scale)

    gray = skimage.transform.resize(gray, (H, W))
    edge_map = skimage.feature.canny(gray)

    # Extract corners
    corner_response: np.ndarray = skimage.feature.corner_moravec(gray) * edge_map

    corners: np.ndarray = skimage.feature.corner_peaks(
        corner_response,
        min_distance=30,
    )

    corners = (corners * (scaling_factor ** scale)).astype(np.int64)
    
    X = corners[:, 1:2]
    Y = corners[:, 0:1]
    
    return np.hstack([X, Y])

def grid_sample(
    seed: int,
    image: np.ndarray,
    camera: Camera,
    base_size: tuple[int, int],
    scale: int,
    num_scales: int,
    scaling_factor: float,
) -> np.ndarray:
    
    patch_size = get_patch_size(base_size, scale, scaling_factor)
    
    grid_x, grid_y = generate_grid(image, patch_size, scale)

    return np.hstack([grid_x[..., None], grid_y[..., None]])

def random_sample(
    seed: int,
    image: np.ndarray,
    camera: Camera,
    base_size: tuple[int, int],
    scale: int,
    num_scales: int,
    scaling_factor: float,
) -> np.ndarray:
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    h, w = get_patch_size(base_size, scale, scaling_factor)
    n = 100
    Y = rng.integers(low=h//2, high=image.shape[0] - h//2, size=(n, 1), dtype=np.int64)
    X = rng.integers(low=w//2, high=image.shape[1] - w//2, size=(n, 1), dtype=np.int64)

    return np.hstack([X, Y])