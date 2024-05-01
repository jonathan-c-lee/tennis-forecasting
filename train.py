"""Train script."""
import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torchvision.io import ImageReadMode

from dataloaders.datasets import TrackNetDataset, BaselineTrajectoryPredictorDataset
from models.tracknet import TrackNet
from models.trajectory_predictor import TrajectoryBaseline
from losses.focal_loss import BinaryFocalLoss


def train(
        model: nn.Module,
        config: dict,
        device: torch.device,
        dataloader: DataLoader,
        optimizer: Optimizer,
        loss_criterion: nn.Module,
        epochs: int,
        model_name: str,
        model_dir: str,
        config_dir: str,
        save_all: bool = True):
    """
    Train model.

    Args:
        model (nn.Module): Model to train.
        config (dict): Model configuration file.
        device (torch.device): Device for model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer for training.
        loss_criterion (nn.Module): Loss criterion for training.
        epochs (int): Number of epochs.
        model_name (str): Name of model for file names.
        model_dir (str): Name of directory to save model in.
        config_dir (str): Name of directory to save model configuration file in.
        save_all (bool): Whether to save each intermediate model.
    """
    model.train()
    train_loss = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        count = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
        train_loss.append(running_loss / count)

        if save_all:
            torch.save(model.state_dict(), f'{model_dir}/{model_name}_{epoch+1}.pt')
            with open(f'{config_dir}/{model_name}_{epoch+1}.yaml', 'w') as outfile:
                config['epochs'] = epoch+1
                config['train loss'] = train_loss
                yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
    torch.save(model.state_dict(), f'{model_dir}/{model_name}.pt')
    with open(f'{config_dir}/{model_name}.yaml', 'w') as outfile:
        config['epochs'] = epochs
        config['train loss'] = train_loss
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    """Training tuning trajectory prediction models."""
    dirname = os.path.dirname(__file__)
    input_frames = (4, 6, 8)
    output_frames = 15
    hidden_dims = (64, 128, 256)
    lstm_layers = (2, 3, 4, 5)
    dropout = (0.0, 0.2, 0.4)

    lr = 2e-2
    epochs = 400
    batch_size = 64

    for input_length in input_frames:
        for hidden_dim in hidden_dims:
            for layers in lstm_layers:
                for drop in dropout:
                    model_name = f'baseline_tuning_{input_length}in_{hidden_dim}hidden_{layers}layers_{int(100*drop)}drop'
                    config = {
                        'name': model_name,
                        'frames_in': input_length,
                        'frames_out': output_frames,
                        'layers': layers,
                        'hidden_size': hidden_dim,
                        'dropoout': drop,
                        'loss': 'MSE loss',
                        'optimizer': 'Adam',
                        'learning rate': lr,
                        'epochs': epochs,
                        'batch_size': batch_size,
                    }

                    datasets = []
                    for i in range(10):
                        data_path = os.path.join(dirname, f'./local_data/game{i+1}/Clip1')
                        dataset = BaselineTrajectoryPredictorDataset(
                            data_path, input_length, output_frames
                        )
                        datasets.append(dataset)
                    train_set = torch.utils.data.ConcatDataset(datasets)
                    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

                    model = TrajectoryBaseline(output_frames, hidden_dim, layers, drop)
                    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
                    model.to(device)

                    loss_criterion = nn.MSELoss()
                    optimizer = Adam(model.parameters(), lr=lr)

                    train(
                        model,
                        config,
                        device,
                        dataloader,
                        optimizer,
                        loss_criterion,
                        epochs,
                        model_name,
                        os.path.join(dirname, f'./trained_models/trajectory'),
                        os.path.join(dirname, f'./configs/trajectory'),
                        save_all=False
                    )
    
    """Training final tracknet model."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 3
    # output_size = 3
    # mode = ImageReadMode.RGB
    # channels = [32, 64, 128, 256]
    # lr = 1e-3
    # epochs = 30
    # batch_size = 8
    # mode_name = 'RGB'

    # model_name = 'tracknet'
    # config = {
    #     'name': model_name,
    #     'shape_in': shape,
    #     'frames_in': input_size,
    #     'frames_out': output_size,
    #     'mode': mode_name,
    #     'channels': channels,
    #     'loss': 'focal loss',
    #     'optimizer': 'Adam',
    #     'learning rate': lr,
    #     'epochs': 0,
    #     'batch_size': batch_size,
    # }

    # data_dictionary = {
    #     'game1': [f'Clip{i+1}' for i in range(13)],
    #     'game2': [f'Clip{i+1}' for i in range(8)],
    #     'game3': [f'Clip{i+1}' for i in range(9)],
    #     'game4': [f'Clip{i+1}' for i in range(7)],
    #     'game5': [f'Clip{i+1}' for i in range(15)],
    #     'game6': [f'Clip{i+1}' for i in range(4)],
    #     'game7': [f'Clip{i+1}' for i in range(9)],
    #     # 'game8': [f'Clip{i+1}' for i in range(9)],
    #     # 'game9': [f'Clip{i+1}' for i in range(9)],
    #     # 'game10': [f'Clip{i+1}' for i in range(12)],
    # }

    # datasets = []
    # for game, clips in data_dictionary.items():
    #     for clip in clips:
    #         data_path = os.path.join(dirname, f'./data/{game}/{clip}')
    #         dataset = TrackNetDataset(
    #             data_path,
    #             shape=shape,
    #             input_size=input_size,
    #             output_size=output_size,
    #             mode=mode
    #         )
    #         datasets.append(dataset)
    # train_set = torch.utils.data.ConcatDataset(datasets)
    # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # model = TrackNet(3*input_size, output_size, channels=channels)
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # loss_criterion = BinaryFocalLoss()
    # optimizer = Adam(model.parameters(), lr=lr)

    # train(
    #     model,
    #     config,
    #     device,
    #     dataloader,
    #     optimizer,
    #     loss_criterion,
    #     epochs,
    #     model_name,
    #     os.path.join(dirname, f'./trained_models/tracknet'),
    #     os.path.join(dirname, f'./configs/tracknet')
    # )

    """Training different sized tracknet models."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 5
    # output_size = 5
    # mode = ImageReadMode.RGB
    # # channels = ([8, 16, 32, 64], [32, 64, 128, 256])
    # # names = ('small', 'large')
    # channels = ([32, 64, 128, 256],)
    # names = ('large',)
    # lr = 2e-3
    # epochs = 30
    # batch_size = 8

    # for name, channel in zip(names, channels):
    #     mode_name = 'GRAY' if mode == ImageReadMode.GRAY else 'RGB'
    #     model_name = f'tracknet{output_size}_{name}'
    #     config = {
    #         'name': model_name,
    #         'shape_in': shape,
    #         'frames_in': input_size,
    #         'frames_out': output_size,
    #         'mode': mode_name,
    #         'channels': channel,
    #         'loss': 'focal loss',
    #         'optimizer': 'Adam',
    #         'learning rate': lr,
    #         'epochs': epochs,
    #         'batch_size': batch_size,
    #     }

    #     datasets = []
    #     for i in range(10):
    #         data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #         dataset = TrackNetDataset(
    #             data_path,
    #             shape=shape,
    #             input_size=input_size,
    #             output_size=output_size,
    #             mode=mode
    #         )
    #         datasets.append(dataset)
    #     train_set = torch.utils.data.ConcatDataset(datasets)
    #     dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    #     multiplier = 1 if mode == ImageReadMode.GRAY else 3
    #     model = TrackNet(multiplier*input_size, output_size, channels=channel)
    #     device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #     model.to(device)

    #     loss_criterion = BinaryFocalLoss()
    #     optimizer = Adam(model.parameters(), lr=lr)

    #     train(
    #         model,
    #         config,
    #         device,
    #         dataloader,
    #         optimizer,
    #         loss_criterion,
    #         epochs,
    #         model_name,
    #         os.path.join(dirname, f'./trained_models/tracknet'),
    #         os.path.join(dirname, f'./configs/tracknet')
    #     )

    """Training tuning tracknet models."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 5
    # output_sizes = (5,)
    # modes = (ImageReadMode.RGB,)
    # lr = 2e-3
    # epochs = 30
    # batch_size = 8

    # for output_size in output_sizes:
    #     for mode in modes:
    #         mode_name = 'GRAY' if mode == ImageReadMode.GRAY else 'RGB'
    #         model_name = f'tracknet_tuning_{input_size}in_{output_size}out_{mode_name}'
    #         config = {
    #             'name': model_name,
    #             'shape_in': shape,
    #             'frames_in': input_size,
    #             'frames_out': output_size,
    #             'mode': mode_name,
    #             'loss': 'focal loss',
    #             'optimizer': 'Adam',
    #             'learning rate': lr,
    #             'epochs': epochs,
    #             'batch_size': batch_size,
    #         }

    #         datasets = []
    #         for i in range(10):
    #             data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #             dataset = TrackNetDataset(
    #                 data_path,
    #                 shape=shape,
    #                 input_size=input_size,
    #                 output_size=output_size,
    #                 mode=mode
    #             )
    #             datasets.append(dataset)
    #         train_set = torch.utils.data.ConcatDataset(datasets)
    #         dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    #         multiplier = 1 if mode == ImageReadMode.GRAY else 3
    #         model = TrackNet(multiplier*input_size, output_size)
    #         device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #         model.to(device)

    #         loss_criterion = BinaryFocalLoss()
    #         optimizer = Adam(model.parameters(), lr=lr)

    #         train(
    #             model,
    #             config,
    #             device,
    #             dataloader,
    #             optimizer,
    #             loss_criterion,
    #             epochs,
    #             model_name,
    #             os.path.join(dirname, f'./trained_models/tracknet'),
    #             os.path.join(dirname, f'./configs/tracknet')
    #         )

