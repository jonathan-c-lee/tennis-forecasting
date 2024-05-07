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
    """
    Dataset for TrackNet model.
    
    Assumes ball tracking annotations are in file called Label.csv with image names in column 0,
    ball x-coordinate in column 2, and ball y-coordinate in column 3.
    """
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
            path (str): Absolute path to dataset.
            shape (Tuple[int, int]): Output image shape as height, width.
            input_size (int): Number of input images for model.
            output_size (int): Number of outputs for model.
            mode (ImageReadMode): Torch image reading mode.
            has_labels (bool): Whether or not the dataset has labels.
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

    @property
    def shape(self):
        """Output image shape."""
        return self._shape

    @property
    def summary(self):
        """Dataset summary."""
        return (f'Image Output Shape: {self._shape}, '
                f'Number of Images: {self._input_size}, '
                f'Number of Outputs: {self._output_size}')

    def __len__(self):
        """Dataset length."""
        return len(self._annotations) - self._input_size

    def __getitem__(self, index: Union[int, torch.Tensor]):
        """
        Dataset indexer.

        Args:
            index (Union[int, torch.Tensor]): Index.
        """
        if torch.is_tensor(index): index = index.tolist()

        image_names = [self._annotations.iloc[index+i, 0] for i in range(self._input_size)]
        images = [read_image(f'{self._path}/{image_name}', self._mode) for image_name in image_names]
        
        raw_shape = (images[0].size(dim=1), images[0].size(dim=2))

        images = torch.cat([self._transform(image) for image in images], dim=0)

        if self._has_labels:
            h_ratio, w_ratio = self._shape[0] / raw_shape[0], self._shape[1] / raw_shape[1]
            start = index + self._input_size - self._output_size
            centers = [
                (int(w_ratio * self._annotations.iloc[start+i, 2]),
                 int(h_ratio * self._annotations.iloc[start+i, 3]))
                for i in range(self._output_size)
            ]
            labels = torch.stack(
                [torch.from_numpy(generate_heatmap(self._shape, center)) for center in centers], dim=0
            )
            return images, labels
        return images


class BaselineTrajectoryPredictorDataset(Dataset):
    """
    Dataset for baseline trajectory prediction model.
    
    Assumes ball tracking annotations are in file called Label.csv with image names in column 0,
    ball x-coordinate in column 2, and ball y-coordinate in column 3.
    """
    def __init__(
            self,
            path: str,
            input_size: int = 4,
            output_size: int = 15,
            mirror: bool = False):
        """
        TrackNet dataset initializer.
        
        Args:
            path (str): Absolute path to dataset.
            input_size (int): Number of input images for model.
            output_size (int): Number of outputs for model.
            mirror (bool): Whether or not to left-right mirror datasets.
        """
        self._input_size = input_size
        self._output_size = output_size

        self._ball_annotations = pd.read_csv(f'{path}/Label.csv').dropna()
        self._ball_annotations = self._ball_annotations.sort_values(by='file name').reset_index(drop=True)

        self._image_shape = read_image(f'{path}/{self._ball_annotations.iloc[0, 0]}').size()
        width, height = self._image_shape[2], self._image_shape[1]

        self._ball_annotations.iloc[:, 2] /= width
        self._ball_annotations.iloc[:, 3] /= height

        if mirror:
            self._ball_annotations.iloc[:, 2] += 2 * (0.5 - self._ball_annotations.iloc[:, 2])

            view = self._ball_annotations.iloc[:, 2]
            self._ball_annotations.iloc[:, 2] = view.mask(view >= 0.9999, 0.0)

    def __len__(self):
        """Dataset length."""
        return len(self._ball_annotations) - self._input_size - self._output_size + 1

    def __getitem__(self, index: Union[int, torch.Tensor]):
        """
        Dataset indexer.

        Args:
            index (Union[int, torch.Tensor]): Index.
        """
        if torch.is_tensor(index): index = index.tolist()

        ball_positions = torch.tensor([
            [float(self._ball_annotations.iloc[index+i, j+2]) for j in range(2)]
            for i in range(self._input_size + self._output_size)
        ])

        inputs = ball_positions[:self._input_size]
        outputs = ball_positions[self._input_size:].view(-1)
        return inputs, outputs


class TrajectoryPredictorDataset(Dataset):
    """
    Dataset for trajectory prediction model.
    
    Assumes ball tracking annotations are in file called Label.csv with image names in column 0,
    ball x-coordinate in column 2, and ball y-coordinate in column 3.
    Assumes player position annotations are in a directory called player_data with csv files with
    image names in column 0 and positions in subsequent columns.
    Assumes player pose annotations are in a directory called player_keypoints with csv files with
    image names in column 0 and positions in subsequent columns.
    """
    def __init__(
            self,
            ball_path: str,
            player_position_path: str,
            player_pose_path: str,
            input_size: int = 4,
            output_size: int = 15,
            mirror: bool = False):
        """
        TrackNet dataset initializer.
        
        Args:
            ball_path (str): Absolute path to ball position dataset.
            player_position_path (str): Absolute path to player position dataset.
            player_pose_path (str): Absolute path to player pose dataset.
            input_size (int): Number of input images for model.
            output_size (int): Number of outputs for model.
            mirror (bool): Whether or not to left-right mirror datasets.
        """
        self._input_size = input_size
        self._output_size = output_size

        self._ball_annotations = pd.read_csv(f'{ball_path}/Label.csv').dropna()

        self._player_poses = pd.read_csv(player_pose_path).fillna(0.0)
        view = self._player_poses.loc[:, self._player_poses.columns != 'image_name']
        self._player_poses.loc[:, self._player_poses.columns != 'image_name'] = view.mask(view < 1e-4, 0.0)

        self._player_positions = pd.read_csv(player_position_path)
        self._player_positions['player_x'] = (self._player_positions['x1'] + self._player_positions['x2']) / 2
        self._player_positions['player_y'] = (self._player_positions['y1'] + self._player_positions['y2']) / 2
        self._player_positions.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)
        self._player_positions = self._player_positions.groupby('image_name').agg(
            player1_x=pd.NamedAgg(column='player_x', aggfunc='first'),
            player1_y=pd.NamedAgg(column='player_y', aggfunc='first'),
            player2_x=pd.NamedAgg(column='player_x', aggfunc='last'),
            player2_y=pd.NamedAgg(column='player_y', aggfunc='last')
        ).reset_index()
        self._player_positions['x_diff'] = (self._player_positions['player1_x'] - self._player_positions['player2_x'])
        self._player_positions['x_diff'] = self._player_positions['x_diff'].abs()
        self._player_positions['y_diff'] = (self._player_positions['player1_y'] - self._player_positions['player2_y'])
        self._player_positions['y_diff'] = self._player_positions['y_diff'].abs()
        self._player_positions['player2_x'].mask(self._player_positions['x_diff'] < 1.0, 0.0, inplace=True)
        self._player_positions['player2_y'].mask(self._player_positions['y_diff'] < 1.0, 0.0, inplace=True)
        self._player_positions.drop(columns=['x_diff', 'y_diff'], inplace=True)

        shared_images = pd.merge(
            left=self._ball_annotations['file name'],
            right=self._player_poses['image_name'],
            how='inner',
            left_on='file name',
            right_on='image_name'
        ).merge(
            self._player_positions,
            how='inner',
            left_on='file name',
            right_on='image_name'
        )['file name'].tolist()

        self._ball_annotations = self._ball_annotations[self._ball_annotations['file name'].isin(shared_images)]
        self._player_poses = self._player_poses[self._player_poses['image_name'].isin(shared_images)]
        self._player_positions = self._player_positions[self._player_positions['image_name'].isin(shared_images)]

        self._ball_annotations = self._ball_annotations.sort_values(by='file name').reset_index(drop=True)
        self._player_poses = self._player_poses.sort_values(by='image_name').reset_index(drop=True)
        self._player_positions = self._player_positions.sort_values(by='image_name').reset_index(drop=True)

        self._image_shape = read_image(f'{ball_path}/{self._ball_annotations.iloc[0, 0]}').size()
        width, height = self._image_shape[2], self._image_shape[1]

        self._ball_annotations.iloc[:, 2] /= width
        self._ball_annotations.iloc[:, 3] /= height
        self._player_poses.iloc[:, 1::2] /= width
        self._player_poses.iloc[:, 2::2] /= height
        self._player_positions.iloc[:, [1, 3]] /= width
        self._player_positions.iloc[:, [2, 4]] /= height

        if mirror:
            self._ball_annotations.iloc[:, 2] += 2 * (0.5 - self._ball_annotations.iloc[:, 2])
            self._player_poses.iloc[:, 1::2] += 2 * (0.5 - self._player_poses.iloc[:, 1::2])
            self._player_positions.iloc[:, 1::2] += 2 * (0.5 - self._player_positions.iloc[:, 1::2])

            view = self._ball_annotations.iloc[:, 2]
            self._ball_annotations.iloc[:, 2] = view.mask(view >= 0.9999, 0.0)
            view = self._player_poses.iloc[:, 1::2]
            self._player_poses.iloc[:, 1::2] = view.mask(view >= 0.9999, 0.0)
            view = self._player_positions.iloc[:, 1::2]
            self._player_positions.iloc[:, 1::2] = view.mask(view >= 0.9999, 0.0)

    def __len__(self):
        """Dataset length."""
        return len(self._ball_annotations) - self._input_size - self._output_size + 1

    def __getitem__(self, index: Union[int, torch.Tensor]):
        """
        Dataset indexer.

        Args:
            index (Union[int, torch.Tensor]): Index.
        """
        if torch.is_tensor(index): index = index.tolist()

        ball_positions = torch.tensor([
            [float(self._ball_annotations.iloc[index+i, j+2]) for j in range(2)]
            for i in range(self._input_size + self._output_size)
        ])

        player_poses = torch.tensor([
            [float(self._player_poses.iloc[index+i, j+1]) for j in range(68)]
            for i in range(self._input_size)
        ])

        player_positions = torch.tensor([
            [float(self._player_positions.iloc[index+i, j+1]) for j in range(4)]
            for i in range(self._input_size)
        ])

        inputs = (ball_positions[:self._input_size], player_positions, player_poses)
        outputs = ball_positions[self._input_size:].view(-1)
        return inputs, outputs


class TrajectoryPredictorPositionOnlyDataset(Dataset):
    """
    Dataset for trajectory prediction model using only ball and player positions.
    
    Assumes ball tracking annotations are in file called Label.csv with image names in column 0,
    ball x-coordinate in column 2, and ball y-coordinate in column 3.
    Assumes player position annotations are in a directory called player_data with csv files with
    image names in column 0 and positions in subsequent columns.
    """
    def __init__(
            self,
            ball_path: str,
            player_position_path: str,
            input_size: int = 4,
            output_size: int = 15,
            mirror: bool = False):
        """
        TrackNet dataset initializer.
        
        Args:
            ball_path (str): Absolute path to ball position dataset.
            player_position_path (str): Absolute path to player position dataset.
            input_size (int): Number of input images for model.
            output_size (int): Number of outputs for model.
            mirror (bool): Whether or not to left-right mirror datasets.
        """
        self._input_size = input_size
        self._output_size = output_size

        self._ball_annotations = pd.read_csv(f'{ball_path}/Label.csv').dropna()

        self._player_positions = pd.read_csv(player_position_path)
        self._player_positions['player_x'] = (self._player_positions['x1'] + self._player_positions['x2']) / 2
        self._player_positions['player_y'] = (self._player_positions['y1'] + self._player_positions['y2']) / 2
        self._player_positions.drop(columns=['x1', 'y1', 'x2', 'y2'], inplace=True)
        self._player_positions = self._player_positions.groupby('image_name').agg(
            player1_x=pd.NamedAgg(column='player_x', aggfunc='first'),
            player1_y=pd.NamedAgg(column='player_y', aggfunc='first'),
            player2_x=pd.NamedAgg(column='player_x', aggfunc='last'),
            player2_y=pd.NamedAgg(column='player_y', aggfunc='last')
        ).reset_index()
        self._player_positions['x_diff'] = (self._player_positions['player1_x'] - self._player_positions['player2_x'])
        self._player_positions['x_diff'] = self._player_positions['x_diff'].abs()
        self._player_positions['y_diff'] = (self._player_positions['player1_y'] - self._player_positions['player2_y'])
        self._player_positions['y_diff'] = self._player_positions['y_diff'].abs()
        self._player_positions['player2_x'].mask(self._player_positions['x_diff'] < 1.0, 0.0, inplace=True)
        self._player_positions['player2_y'].mask(self._player_positions['y_diff'] < 1.0, 0.0, inplace=True)
        self._player_positions.drop(columns=['x_diff', 'y_diff'], inplace=True)

        shared_images = pd.merge(
            left=self._ball_annotations['file name'],
            right=self._player_positions['image_name'],
            how='inner',
            left_on='file name',
            right_on='image_name'
        )['file name'].tolist()

        self._ball_annotations = self._ball_annotations[self._ball_annotations['file name'].isin(shared_images)]
        self._player_positions = self._player_positions[self._player_positions['image_name'].isin(shared_images)]

        self._ball_annotations = self._ball_annotations.sort_values(by='file name').reset_index(drop=True)
        self._player_positions = self._player_positions.sort_values(by='image_name').reset_index(drop=True)

        self._image_shape = read_image(f'{ball_path}/{self._ball_annotations.iloc[0, 0]}').size()
        width, height = self._image_shape[2], self._image_shape[1]

        self._ball_annotations.iloc[:, 2] /= width
        self._ball_annotations.iloc[:, 3] /= height
        self._player_positions.iloc[:, [1, 3]] /= width
        self._player_positions.iloc[:, [2, 4]] /= height

        if mirror:
            self._ball_annotations.iloc[:, 2] += 2 * (0.5 - self._ball_annotations.iloc[:, 2])
            self._player_positions.iloc[:, 1::2] += 2 * (0.5 - self._player_positions.iloc[:, 1::2])

            view = self._ball_annotations.iloc[:, 2]
            self._ball_annotations.iloc[:, 2] = view.mask(view >= 0.9999, 0.0)
            view = self._player_positions.iloc[:, 1::2]
            self._player_positions.iloc[:, 1::2] = view.mask(view >= 0.9999, 0.0)

    def __len__(self):
        """Dataset length."""
        return len(self._ball_annotations) - self._input_size - self._output_size + 1

    def __getitem__(self, index: Union[int, torch.Tensor]):
        """
        Dataset indexer.

        Args:
            index (Union[int, torch.Tensor]): Index.
        """
        if torch.is_tensor(index): index = index.tolist()

        ball_positions = torch.tensor([
            [float(self._ball_annotations.iloc[index+i, j+2]) for j in range(2)]
            for i in range(self._input_size + self._output_size)
        ])

        player_positions = torch.tensor([
            [float(self._player_positions.iloc[index+i, j+1]) for j in range(4)]
            for i in range(self._input_size)
        ])

        inputs = (ball_positions[:self._input_size], player_positions)
        outputs = ball_positions[self._input_size:].view(-1)
        return inputs, outputs


if __name__ == '__main__':
    # testing the TrackNetDataset class
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../data/game1/Clip1')
    dataset = TrackNetDataset(path, shape=(288, 512))
    print('Sample Input Shape:', dataset[0][0].size())
    print('Sample Label Shape:', dataset[0][1].size())
    print(dataset.summary)