"""Datasets for models."""
import os
from typing import Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import ImageReadMode, read_image
from dataloaders.heatmap import generate_heatmap


class TrackNetDataset(Dataset):
    """Dataset for TrackNet model."""
    def __init__(
            self,
            path: str,
            shape: Tuple[int, int],
            input_size: int = 3,
            output_size: int = 3,
            mode: ImageReadMode = ImageReadMode.RGB,
            has_labels: bool = True):
        """
        TrackNet dataset initializer.
        
        Args:
            path (str): absolute path to dataset
            shape (Tuple[int, int]): output image shape as height, width
            input_size (int): number of input images for model
            output_size (int): number of outputs for model
            mode (ImageReadMode): torch image reading mode
            has_labels (bool): whether or not dataset has labels
        """
        if input_size < output_size:
            raise ValueError('Cannot have more outputs than inputs')

        self._path = path
        self._annotations = pd.read_csv(f'{path}/Label.csv').fillna(-1)
        self._shape = shape
        self._input_size = input_size
        self._output_size = output_size
        self._transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(shape)
        ])
        self._mode = mode
        self._has_labels = has_labels

    def __len__(self):
        """Dataset length."""
        return len(self._annotations) - self._input_size

    def __getitem__(self, index: Union[int, torch.Tensor]):
        """
        Dataset indexer.

        Args:
            index (Union[int, torch.Tensor]): index
        """
        if torch.is_tensor(index): index = index.tolist()

        image_names = [self._annotations.iloc[index+i, 0] for i in range(self._input_size)]
        images = [read_image(f'{self._path}/{image_name}', self._mode) for image_name in image_names]
        images = [self._transform(image) for image in images]
        images = torch.cat(images, dim=0)

        if self._has_labels:
            start = index + self._input_size - self._output_size
            centers = [
                (self._annotations.iloc[start+i, 1], self._annotations.iloc[start+i, 2])
                for i in range(self._output_size)
            ]
            labels = [torch.from_numpy(generate_heatmap(self._shape, center)) for center in centers]
            labels = torch.stack(labels, dim=0)
            return images, labels
        return images


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../data/game1/Clip1')
    dataset = TrackNetDataset(path, shape=(288, 512))
    print('Sample Input Shape:', dataset[0][0].size())
    print('Sample Label Shape:', dataset[0][1].size())