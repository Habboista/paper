"""
Eigen split has more samples than the official kitti split for some reason,
this script saves those kitti filenames.
"""

import os
from tqdm import tqdm
from src.data.kitti.dataset import KITTIDataset

def produce_split(mode: str) -> None:
    data = KITTIDataset(mode, from_velodyne=False)
    num_valid_samples = len(data)

    with open(os.path.join('src', 'data', 'kitti', 'split', 'kitti_' + mode + '_files.txt'), 'w') as f:
        for i in tqdm(range(len(data))):
            try:
                data[i]
            except FileNotFoundError:
                num_valid_samples -= 1
            else:
                f.write(f"{data.filenames[i]}\n")

    print(f"{mode.upper()} Eigen split has {num_valid_samples} / {len(data)} = {num_valid_samples / len(data):.2f} % files annotated with dense(er) groundtruth.")

def main():
    produce_split('test')
    produce_split('val')
    produce_split('train')

if __name__ == '__main__':
    main()