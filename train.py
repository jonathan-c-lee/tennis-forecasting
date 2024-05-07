"""Train script."""
import os

os.environ["OMP_NUM_THREADS"] = "30"
os.environ["MKL_NUM_THREADS"] = "30"

import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam, lr_scheduler
from torchvision.io import ImageReadMode

from dataloaders.datasets import *
from models.tracknet import TrackNet
from models.trajectory_predictor import *
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

        if save_all and (epoch+1) % 20 == 0:
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
    """Training final trajectory prediction position only model."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # position_dim = 4
    # hidden_dims = 256
    # lstm_layers = 3
    # dropout = 0.2

    # lr = 1e-3
    # weight_decay = 1e-4
    # epochs = 400
    # batch_size = 32

    # model_name = f'position_trajectory_predictor'
    # config = {
    #     'name': model_name,
    #     'frames_in': input_frames,
    #     'frames_out': output_frames,
    #     'position_dim': position_dim,
    #     'layers': lstm_layers,
    #     'hidden_size': hidden_dims,
    #     'dropout': dropout,
    #     'loss': 'MSE loss',
    #     'optimizer': 'Adam',
    #     'learning rate': lr,
    #     'weight decay': weight_decay,
    #     'epochs': epochs,
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
    #         ball_path = os.path.join(dirname, f'./data/{game}/{clip}')
    #         player_position_path = os.path.join(dirname, f'./player_data/{game}_{clip}_player_positions.csv')
    #         dataset = TrajectoryPredictorPositionOnlyDataset(
    #             ball_path, player_position_path, input_frames, output_frames
    #         )
    #         datasets.append(dataset)
    #         mirror_dataset = TrajectoryPredictorPositionOnlyDataset(
    #             ball_path, player_position_path, input_frames, output_frames, mirror=True
    #         )
    #         datasets.append(mirror_dataset)
    # train_set = torch.utils.data.ConcatDataset(datasets)
    # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    # model = PositionTrajectoryPredictor(
    #     output_frames,
    #     position_dim,
    #     hidden_dims,
    #     lstm_layers,
    #     dropout
    # )
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # loss_criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.25, patience=10, threshold=1e-4
    # )

    # model.train()
    # train_loss = []
    # for epoch in tqdm(range(epochs)):
    #     running_loss = 0.0
    #     count = 0
    #     for inputs, labels in dataloader:
    #         ball_positions = inputs[0].to(device)
    #         player_positions = inputs[1].to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(ball_positions, player_positions)
    #         loss = loss_criterion(outputs, labels)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         running_loss += loss.item()
    #         count += 1
    #     train_loss.append(running_loss / count)
    #     scheduler.step(running_loss / count)
    #     if (epoch+1) % 20 == 0:
    #         torch.save(model.state_dict(), f'./trained_models/position_trajectory/{model_name}_{epoch+1}.pt')
    #         torch.save(optimizer.state_dict(), f'./trained_models/optimizer/{model_name}_{epoch+1}_optimizer.pt')
    #         with open(f'./configs/position_trajectory/{model_name}_{epoch+1}.yaml', 'w') as outfile:
    #             config['epochs'] = epoch+1
    #             config['train loss'] = train_loss
    #             yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
    #     print(train_loss[-1])
    # torch.save(model.state_dict(), f'./trained_models/position_trajectory/{model_name}.pt')
    # torch.save(optimizer.state_dict(), f'./trained_models/optimizer/{model_name}_optimizer.pt')
    # with open(f'./configs/position_trajectory/{model_name}.yaml', 'w') as outfile:
    #     config['epochs'] = epochs
    #     config['train loss'] = train_loss
    #     yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training tuning trajectory prediction position only models."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # position_dim = 4
    # hidden_dims = (64, 128, 256)
    # lstm_layers = (2, 3, 4)
    # dropout = 0.2

    # lr = 1e-3
    # weight_decay = 1e-4
    # epochs = 40
    # batch_size = (32, 64, 128)

    # for hidden_dim in hidden_dims:
    #     for layers in lstm_layers:
    #         for batch in batch_size:
    #             model_name = f'tuning_{hidden_dim}hidden_{layers}layers_{batch}batch'
    #             config = {
    #                 'name': model_name,
    #                 'frames_in': input_frames,
    #                 'frames_out': output_frames,
    #                 'position_dim': position_dim,
    #                 'layers': layers,
    #                 'hidden_size': hidden_dim,
    #                 'dropout': dropout,
    #                 'loss': 'MSE loss',
    #                 'optimizer': 'Adam',
    #                 'learning rate': lr,
    #                 'weight decay': weight_decay,
    #                 'epochs': epochs,
    #                 'batch_size': batch,
    #             }

    #             datasets = []
    #             for i in range(7):
    #                 ball_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #                 player_position_path = os.path.join(dirname, f'./player_data/game{i+1}_Clip1_player_positions.csv')
    #                 dataset = TrajectoryPredictorPositionOnlyDataset(
    #                     ball_path, player_position_path, input_frames, output_frames
    #                 )
    #                 datasets.append(dataset)
    #             train_set = torch.utils.data.ConcatDataset(datasets)
    #             dataloader = DataLoader(train_set, batch_size=batch, shuffle=True)

    #             model = PositionTrajectoryPredictor(
    #                 output_frames,
    #                 position_dim,
    #                 hidden_dim,
    #                 layers,
    #                 dropout
    #             )
    #             device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #             model.to(device)

    #             loss_criterion = nn.MSELoss()
    #             optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    #             model.train()
    #             train_loss = []
    #             for epoch in tqdm(range(epochs)):
    #                 running_loss = 0.0
    #                 count = 0
    #                 for inputs, labels in dataloader:
    #                     ball_positions = inputs[0].to(device)
    #                     player_positions = inputs[1].to(device)
    #                     labels = labels.to(device)

    #                     optimizer.zero_grad()
    #                     outputs = model(ball_positions, player_positions)
    #                     loss = loss_criterion(outputs, labels)
    #                     loss.backward()
    #                     optimizer.step()

    #                     running_loss += loss.item()
    #                     count += 1
    #                 train_loss.append(running_loss / count)
    #             torch.save(model.state_dict(), f'./trained_models/position_trajectory/{model_name}.pt')
    #             with open(f'./configs/position_trajectory/{model_name}.yaml', 'w') as outfile:
    #                 config['epochs'] = epochs
    #                 config['train loss'] = train_loss
    #                 yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training final trajectory prediction model."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # position_dim = 4
    # pose_size = 68
    # pose_dims = 16
    # hidden_dims = 256
    # lstm_layers = 3
    # dropout = 0.2

    # lr = 1e-3
    # weight_decay = 1e-4
    # epochs = 400
    # batch_size = 32

    # model_name = f'trajectory_predictor_0'
    # config = {
    #     'name': model_name,
    #     'frames_in': input_frames,
    #     'frames_out': output_frames,
    #     'position_dim': position_dim,
    #     'pose_size': pose_size,
    #     'pose_dim': pose_dims,
    #     'layers': lstm_layers,
    #     'hidden_size': hidden_dims,
    #     'dropout': dropout,
    #     'loss': 'MSE loss',
    #     'optimizer': 'Adam',
    #     'learning rate': lr,
    #     'weight decay': weight_decay,
    #     'epochs': epochs,
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
    #         ball_path = os.path.join(dirname, f'./data/{game}/{clip}')
    #         player_position_path = os.path.join(dirname, f'./player_data/{game}_{clip}_player_positions.csv')
    #         player_pose_path = os.path.join(dirname, f'./player_keypoints/{game}_{clip}_keypoints.csv')
    #         dataset = TrajectoryPredictorDataset(
    #             ball_path, player_position_path, player_pose_path, input_frames, output_frames
    #         )
    #         datasets.append(dataset)
    #         mirror_dataset = TrajectoryPredictorDataset(
    #             ball_path, player_position_path, player_pose_path, input_frames, output_frames, mirror=True
    #         )
    #         datasets.append(mirror_dataset)
    # train_set = torch.utils.data.ConcatDataset(datasets)
    # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    # model = TrajectoryPredictor(
    #     output_frames,
    #     position_dim,
    #     pose_size,
    #     pose_dims,
    #     hidden_dims,
    #     lstm_layers,
    #     dropout
    # )
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # loss_criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.25, patience=10, threshold=1e-4
    # )

    # model.train()
    # train_loss = []
    # for epoch in tqdm(range(epochs)):
    #     running_loss = 0.0
    #     count = 0
    #     for inputs, labels in dataloader:
    #         ball_positions = inputs[0].to(device)
    #         player_positions = inputs[1].to(device)
    #         player_poses = inputs[2].to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(ball_positions, player_positions, player_poses)
    #         loss = loss_criterion(outputs, labels)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         running_loss += loss.item()
    #         count += 1
    #     train_loss.append(running_loss / count)
    #     scheduler.step(running_loss / count)
    #     if (epoch+1) % 20 == 0:
    #         torch.save(model.state_dict(), f'./trained_models/trajectory/{model_name}_{epoch+1}.pt')
    #         torch.save(optimizer.state_dict(), f'./trained_models/optimizer/{model_name}_{epoch+1}_optimizer.pt')
    #         with open(f'./configs/trajectory/{model_name}_{epoch+1}.yaml', 'w') as outfile:
    #             config['epochs'] = epoch+1
    #             config['train loss'] = train_loss
    #             yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
    #     print(train_loss[-1])
    # torch.save(model.state_dict(), f'./trained_models/trajectory/{model_name}.pt')
    # torch.save(optimizer.state_dict(), f'./trained_models/optimizer/{model_name}_optimizer.pt')
    # with open(f'./configs/trajectory/{model_name}.yaml', 'w') as outfile:
    #     config['epochs'] = epochs
    #     config['train loss'] = train_loss
    #     yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training tuning trajectory prediction models."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # position_dim = 4
    # pose_size = 68
    # pose_dims = (2, 16, 68)
    # hidden_dims = (128, 256)
    # lstm_layers = (2, 3)
    # dropout = 0.2

    # lr = 1e-3
    # weight_decay = 1e-4
    # epochs = 20
    # batch_size = (32, 64, 128)

    # for pose_length in pose_dims:
    #     for hidden_dim in hidden_dims:
    #         for layers in lstm_layers:
    #             for batch in batch_size:
    #                 model_name = f'tuning_{pose_length}pose_{hidden_dim}hidden_{layers}layers_{batch}batch'
    #                 config = {
    #                     'name': model_name,
    #                     'frames_in': input_frames,
    #                     'frames_out': output_frames,
    #                     'position_dim': position_dim,
    #                     'pose_size': pose_size,
    #                     'pose_dim': pose_length,
    #                     'layers': layers,
    #                     'hidden_size': hidden_dim,
    #                     'dropout': dropout,
    #                     'loss': 'MSE loss',
    #                     'optimizer': 'Adam',
    #                     'learning rate': lr,
    #                     'weight decay': weight_decay,
    #                     'epochs': epochs,
    #                     'batch_size': batch,
    #                 }

    #                 datasets = []
    #                 for i in range(7):
    #                     ball_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #                     player_position_path = os.path.join(dirname, f'./player_data/game{i+1}_Clip1_player_positions.csv')
    #                     player_pose_path = os.path.join(dirname, f'./player_keypoints/game{i+1}_Clip1_keypoints.csv')
    #                     dataset = TrajectoryPredictorDataset(
    #                         ball_path, player_position_path, player_pose_path, input_frames, output_frames
    #                     )
    #                     datasets.append(dataset)
    #                 train_set = torch.utils.data.ConcatDataset(datasets)
    #                 dataloader = DataLoader(train_set, batch_size=batch, shuffle=True)

    #                 model = TrajectoryPredictor(
    #                     output_frames,
    #                     position_dim,
    #                     pose_size,
    #                     pose_length,
    #                     hidden_dim,
    #                     layers,
    #                     dropout
    #                 )
    #                 device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #                 model.to(device)

    #                 loss_criterion = nn.MSELoss()
    #                 optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    #                 model.train()
    #                 train_loss = []
    #                 for epoch in tqdm(range(epochs)):
    #                     running_loss = 0.0
    #                     count = 0
    #                     for inputs, labels in dataloader:
    #                         ball_positions = inputs[0].to(device)
    #                         player_positions = inputs[1].to(device)
    #                         player_poses = inputs[2].to(device)
    #                         labels = labels.to(device)

    #                         optimizer.zero_grad()
    #                         outputs = model(ball_positions, player_positions, player_poses)
    #                         loss = loss_criterion(outputs, labels)
    #                         loss.backward()
    #                         optimizer.step()

    #                         running_loss += loss.item()
    #                         count += 1
    #                     train_loss.append(running_loss / count)
    #                 torch.save(model.state_dict(), f'./trained_models/trajectory/{model_name}.pt')
    #                 with open(f'./configs/trajectory/{model_name}.yaml', 'w') as outfile:
    #                     config['epochs'] = epochs
    #                     config['train loss'] = train_loss
    #                     yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training final baseline trajectory prediction model."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # hidden_dim = 64
    # lstm_layers = 2
    # dropout = 0.0

    # lr = 1e-3
    # weight_decay = 1e-4
    # epochs = 400
    # batch_size = 64

    # model_name = f'baseline_trajectory_predictor'
    # config = {
    #     'name': model_name,
    #     'frames_in': input_frames,
    #     'frames_out': output_frames,
    #     'layers': lstm_layers,
    #     'hidden_size': hidden_dim,
    #     'dropout': dropout,
    #     'loss': 'MSE loss',
    #     'optimizer': 'Adam',
    #     'learning rate': lr,
    #     'weight decay': weight_decay,
    #     'epochs': epochs,
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
    #         ball_path = os.path.join(dirname, f'./data/{game}/{clip}')
    #         dataset = BaselineTrajectoryPredictorDataset(
    #             ball_path, input_frames, output_frames
    #         )
    #         datasets.append(dataset)
    #         mirror_dataset = BaselineTrajectoryPredictorDataset(
    #             ball_path, input_frames, output_frames, mirror=True
    #         )
    #         datasets.append(mirror_dataset)
    # train_set = torch.utils.data.ConcatDataset(datasets)
    # dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # model = TrajectoryBaseline(output_frames, hidden_dim, lstm_layers, dropout)
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # loss_criterion = nn.MSELoss()
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    # model.train()
    # train_loss = []
    # for epoch in tqdm(range(epochs)):
    #     running_loss = 0.0
    #     count = 0
    #     for inputs, labels in dataloader:
    #         ball_positions = inputs.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(ball_positions)
    #         loss = loss_criterion(outputs, labels)
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         running_loss += loss.item()
    #         count += 1
    #     train_loss.append(running_loss / count)
    #     if (epoch+1) % 20 == 0:
    #         torch.save(model.state_dict(), f'./trained_models/baseline_trajectory/{model_name}_{epoch+1}.pt')
    #         with open(f'./configs/baseline_trajectory/{model_name}_{epoch+1}.yaml', 'w') as outfile:
    #             config['epochs'] = epoch+1
    #             config['train loss'] = train_loss
    #             yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
    #     print(train_loss[-1])
    # torch.save(model.state_dict(), f'./trained_models/baseline_trajectory/{model_name}.pt')
    # with open(f'./configs/baseline_trajectory/{model_name}.yaml', 'w') as outfile:
    #     config['epochs'] = epochs
    #     config['train loss'] = train_loss
    #     yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training tuning baseline trajectory prediction models."""
    # torch.set_num_threads(30)
    # torch.set_num_interop_threads(30)
    # dirname = os.path.dirname(__file__)
    # input_frames = (4, 6, 8)
    # output_frames = 15
    # hidden_dims = (64, 128, 256)
    # lstm_layers = (2, 3, 4, 5)
    # dropout = (0.0, 0.2, 0.4)

    # lr = 2e-3
    # weight_decay = 1e-4
    # epochs = 100
    # batch_size = 32

    # for input_length in input_frames:
    #     for hidden_dim in hidden_dims:
    #         for layers in lstm_layers:
    #             for drop in dropout:
    #                 model_name = f'baseline_tuning_{input_length}in_{hidden_dim}hidden_{layers}layers_{int(100*drop)}drop'
    #                 config = {
    #                     'name': model_name,
    #                     'frames_in': input_length,
    #                     'frames_out': output_frames,
    #                     'layers': layers,
    #                     'hidden_size': hidden_dim,
    #                     'dropout': drop,
    #                     'loss': 'MSE loss',
    #                     'optimizer': 'Adam',
    #                     'learning rate': lr,
    #                     'weight decay': weight_decay,
    #                     'epochs': epochs,
    #                     'batch_size': batch_size,
    #                 }

    #                 datasets = []
    #                 for i in range(10):
    #                     data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #                     dataset = BaselineTrajectoryPredictorDataset(
    #                         data_path, input_length, output_frames
    #                     )
    #                     datasets.append(dataset)
    #                 train_set = torch.utils.data.ConcatDataset(datasets)
    #                 dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    #                 model = TrajectoryBaseline(output_frames, hidden_dim, layers, drop)
    #                 device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #                 model.to(device)

    #                 loss_criterion = nn.MSELoss()
    #                 optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    #                 train(
    #                     model,
    #                     config,
    #                     device,
    #                     dataloader,
    #                     optimizer,
    #                     loss_criterion,
    #                     epochs,
    #                     model_name,
    #                     os.path.join(dirname, f'./trained_models/trajectory'),
    #                     os.path.join(dirname, f'./configs/trajectory'),
    #                     save_all=False
    #                 )
    
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

