import os

import numpy as np
import skimage
from tqdm import tqdm

from proposed.train_dataset import TrainDataset
from raw_data.kitti_dataset import KITTIDataset

def main():
    train_data = TrainDataset()
    raw_data = KITTIDataset('test', center_crop=False)
    for idx, (image, _, _) in tqdm(enumerate(raw_data), total=len(raw_data)):
        edge_map: np.ndarray = train_data.compute_edge_map(image, scale=4)

        if not edge_map.any():
            print(idx)
        # Getting the name of the filename (code from KITTIDataset __getitem__ and load_image)
        line = raw_data.filenames[idx].split(' ')

        if len(line) != 3:
            raise ValueError(f"line {idx} does not contain 3 fields")

        folder, frame_index, side = line

        fn = f"edge_{int(frame_index):010d}.png"
        if side not in ["l", "r"]:
            raise ValueError(f"Unknown side {side}")
        filename = os.path.join(
            raw_data.data_path,
            folder,
            "image_02" if side == "l" else "image_03",
            "data",
            fn,
        )

        # Save the edge map
        skimage.io.imsave(filename, edge_map, check_contrast=False)

if __name__ == "__main__":
    main()