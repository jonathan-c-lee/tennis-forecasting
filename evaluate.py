"""Evaluation script."""
import os
import torch
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode

from dataloaders.datasets import TrackNetDataset, BaselineTrajectoryPredictorDataset, TrajectoryPredictorDataset
from models.tracknet import TrackNet
from models.trajectory_predictor import TrajectoryBaseline, TrajectoryPredictor
from metrics.evaluator import evaluate_tracknet


if __name__ == '__main__':
    """Evaluating tuning trajectory prediction models."""
    torch.set_num_threads(30)
    torch.set_num_interop_threads(30)
    dirname = os.path.dirname(__file__)
    input_frames = 6
    output_frames = 15
    position_dim = 4
    pose_size = 68
    pose_dims = (2, 16, 68)
    hidden_dims = (128, 256)
    lstm_layers = (2, 3, 4)
    dropout = (0.0, 0.2)

    train_list = []
    test_list = []
    for pose_length in pose_dims:
        for hidden_dim in hidden_dims:
            for layers in lstm_layers:
                for drop in dropout:
                    model_name = f'tuning_{pose_length}pose_{hidden_dim}hidden_{layers}layers_{int(100*drop)}drop'

                    train_datasets = []
                    for i in range(7):
                        ball_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
                        player_position_path = os.path.join(dirname, f'./player_data/game{i+1}_Clip1_player_positions.csv')
                        player_pose_path = os.path.join(dirname, f'./player_keypoints/game{i+1}_Clip1_keypoints.csv')
                        dataset = TrajectoryPredictorDataset(
                            ball_path, player_position_path, player_pose_path, input_frames, output_frames
                        )
                        train_datasets.append(dataset)
                    train_set = torch.utils.data.ConcatDataset(train_datasets)
                    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

                    test_datasets = []
                    for i in range(7):
                        ball_path = os.path.join(dirname, f'./data/game{i+1}/Clip2')
                        player_position_path = os.path.join(dirname, f'./player_data/game{i+1}_Clip2_player_positions.csv')
                        player_pose_path = os.path.join(dirname, f'./player_keypoints/game{i+1}_Clip2_keypoints.csv')
                        dataset = TrajectoryPredictorDataset(
                            ball_path, player_position_path, player_pose_path, input_frames, output_frames
                        )
                        test_datasets.append(dataset)
                    test_set = torch.utils.data.ConcatDataset(test_datasets)
                    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

                    model = TrajectoryPredictor(
                        output_frames,
                        position_dim,
                        pose_size,
                        pose_length,
                        hidden_dim,
                        layers,
                        drop
                    )
                    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
                    model.load_state_dict(
                        torch.load(os.path.join(dirname, f'./trained_models/trajectory/{model_name}.pt'), map_location=torch.device(device))
                    )
                    model.to(device)
                    model.eval()

                    loss_criterion = torch.nn.MSELoss()

                    running_loss = 0.0
                    count = 0
                    for inputs, labels in train_loader:
                        ball_positions = inputs[0].to(device)
                        player_positions = inputs[1].to(device)
                        player_poses = inputs[2].to(device)
                        labels = labels.to(device)

                        outputs = model(ball_positions, player_positions, player_poses)
                        loss = loss_criterion(outputs, labels)

                        running_loss += loss.item()
                        count += 1
                    print(model_name)
                    print('Train MSE:', running_loss / count)
                    train_list.append((running_loss / count, model_name))

                    running_loss = 0.0
                    count = 0
                    for inputs, labels in test_loader:
                        ball_positions = inputs[0].to(device)
                        player_positions = inputs[1].to(device)
                        player_poses = inputs[2].to(device)
                        labels = labels.to(device)

                        outputs = model(ball_positions, player_positions, player_poses)
                        loss = loss_criterion(outputs, labels)

                        running_loss += loss.item()
                        count += 1
                    print('Test MSE:', running_loss / count)
                    test_list.append((running_loss / count, model_name))
    
    train_list.sort(key = lambda x: x[0])
    test_list.sort(key = lambda x: x[0])

    print('Train')
    print(train_list[:10])
    print('Test')
    print(test_list[:10])

    """Evaluating final trajectory prediction baseline."""
    # dirname = os.path.dirname(__file__)
    # input_frames = 6
    # output_frames = 15
    # hidden_dim = 64
    # lstm_layers = 2
    # dropout = 0.0

    # data_dictionary = {
    #     # 'game1': [f'Clip{i+1}' for i in range(13)],
    #     # 'game2': [f'Clip{i+1}' for i in range(8)],
    #     # 'game3': [f'Clip{i+1}' for i in range(9)],
    #     # 'game4': [f'Clip{i+1}' for i in range(7)],
    #     # 'game5': [f'Clip{i+1}' for i in range(15)],
    #     # 'game6': [f'Clip{i+1}' for i in range(4)],
    #     # 'game7': [f'Clip{i+1}' for i in range(9)],
    #     'game8': [f'Clip{i+1}' for i in range(9)],
    #     'game9': [f'Clip{i+1}' for i in range(9)],
    #     'game10': [f'Clip{i+1}' for i in range(12)],
    # }

    # datasets = []
    # for game, clips in data_dictionary.items():
    #     for clip in clips:
    #         data_path = os.path.join(dirname, f'./data/{game}/{clip}')
    #         dataset = BaselineTrajectoryPredictorDataset(
    #             data_path, input_frames, output_frames
    #         )
    #         datasets.append(dataset)
    # test_set = torch.utils.data.ConcatDataset(datasets)
    # dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    # test_list = []
    # for i in range(50):
    #     model_name = f'baseline_trajectory_predictor_{20*(i+1)}'
    #     model = TrajectoryBaseline(output_frames, hidden_dim, lstm_layers, dropout)
    #     device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #     model.load_state_dict(
    #         torch.load(os.path.join(dirname, f'./trained_models/trajectory/{model_name}.pt'), map_location=torch.device(device))
    #     )
    #     model.to(device)
    #     model.eval()

    #     loss_criterion = torch.nn.MSELoss()

    #     running_loss = 0.0
    #     count = 0
    #     for inputs, labels in dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         outputs = model(inputs)
    #         loss = loss_criterion(outputs, labels)

    #         running_loss += loss.item()
    #         count += 1
    #     print(model_name)
    #     print('Test MSE:', running_loss / count)
    #     test_list.append(running_loss / count)
    # print(test_list)

    """Evaluating tuning baseline trajectory prediction models."""
    # dirname = os.path.dirname(__file__)
    # input_frames = (4, 6, 8)
    # output_frames = 15
    # hidden_dims = (64, 128, 256)
    # lstm_layers = (2, 3, 4, 5)
    # dropout = (0.0, 0.2, 0.4)

    # train_list = []
    # test_list = []
    # for input_length in input_frames:
    #     for hidden_dim in hidden_dims:
    #         for layers in lstm_layers:
    #             for drop in dropout:
    #                 model_name = f'baseline_tuning_{input_length}in_{hidden_dim}hidden_{layers}layers_{int(100*drop)}drop'

    #                 train_datasets = []
    #                 for i in range(10):
    #                     data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #                     dataset = BaselineTrajectoryPredictorDataset(
    #                         data_path, input_length, output_frames
    #                     )
    #                     train_datasets.append(dataset)
    #                 train_set = torch.utils.data.ConcatDataset(train_datasets)
    #                 train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    #                 test_datasets = []
    #                 for i in range(10):
    #                     data_path = os.path.join(dirname, f'./data/game{i+1}/Clip2')
    #                     dataset = BaselineTrajectoryPredictorDataset(
    #                         data_path, input_length, output_frames
    #                     )
    #                     test_datasets.append(dataset)
    #                 test_set = torch.utils.data.ConcatDataset(test_datasets)
    #                 test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    #                 model = TrajectoryBaseline(output_frames, hidden_dim, layers, drop)
    #                 device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #                 model.load_state_dict(
    #                     torch.load(os.path.join(dirname, f'./trained_models/trajectory/{model_name}.pt'), map_location=torch.device(device))
    #                 )
    #                 model.to(device)
    #                 model.eval()

    #                 loss_criterion = torch.nn.MSELoss()

    #                 running_loss = 0.0
    #                 count = 0
    #                 for inputs, labels in train_loader:
    #                     inputs, labels = inputs.to(device), labels.to(device)

    #                     outputs = model(inputs)
    #                     loss = loss_criterion(outputs, labels)

    #                     running_loss += loss.item()
    #                     count += 1
    #                 print(model_name)
    #                 print('Train MSE:', running_loss / count)
    #                 train_list.append((running_loss / count, model_name))

    #                 running_loss = 0.0
    #                 count = 0
    #                 for inputs, labels in test_loader:
    #                     inputs, labels = inputs.to(device), labels.to(device)

    #                     outputs = model(inputs)
    #                     loss = loss_criterion(outputs, labels)

    #                     running_loss += loss.item()
    #                     count += 1
    #                 print('Test MSE:', running_loss / count)
    #                 test_list.append((running_loss / count, model_name))
    
    # train_list.sort(key = lambda x: x[0])
    # test_list.sort(key = lambda x: x[0])

    # print('Train')
    # print(train_list[:10])
    # print('Test')
    # print(test_list[:10])

    """Evaluating tuning models."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 5
    # output_sizes = (3, 5)
    # modes = (ImageReadMode.GRAY, ImageReadMode.RGB)

    # for output_size in output_sizes:
    #     for mode in modes:
    #         mode_name = 'GRAY' if mode == ImageReadMode.GRAY else 'RGB'
    #         model_name = f'tracknet_tuning_{input_size}in_{output_size}out_{mode_name}'

    #         train_datasets = []
    #         for i in range(10):
    #             data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #             dataset = TrackNetDataset(
    #                 data_path,
    #                 shape=shape,
    #                 input_size=input_size,
    #                 output_size=output_size,
    #                 mode=mode
    #             )
    #             train_datasets.append(dataset)
    #         train_set = torch.utils.data.ConcatDataset(train_datasets)
    #         train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    #         test_datasets = []
    #         for i in range(10):
    #             data_path = os.path.join(dirname, f'./data/game{i+1}/Clip2')
    #             dataset = TrackNetDataset(
    #                 data_path,
    #                 shape=shape,
    #                 input_size=input_size,
    #                 output_size=output_size,
    #                 mode=mode
    #             )
    #             test_datasets.append(dataset)
    #         test_set = torch.utils.data.ConcatDataset(test_datasets)
    #         test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    #         multiplier = 1 if mode == ImageReadMode.GRAY else 3
    #         model = TrackNet(multiplier*input_size, output_size)
    #         device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #         model.load_state_dict(
    #             torch.load(os.path.join(dirname, f'./trained_models/tracknet/{model_name}.pt'), map_location=torch.device(device))
    #         )
    #         model.to(device)
    #         model.eval()

    #         train_statistics, train_scores = evaluate_tracknet(model, train_loader, device)
    #         test_statistics, test_scores = evaluate_tracknet(model, test_loader, device)
    #         print(model_name)
    #         print('Train Performance')
    #         print(train_statistics)
    #         print(train_scores)
    #         print('Test Performance')
    #         print(test_statistics)
    #         print(test_scores)

    """Evaluating different sized models."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 5
    # output_size = 5
    # mode = ImageReadMode.RGB
    # channels = ([8, 16, 32, 64], [16, 32, 64, 128], [32, 64, 128, 256])
    # names = ('small', 'medium', 'large')

    # for name, channel in zip(names, channels):
    #     model_name = f'tracknet{output_size}_{name}'

    #     train_datasets = []
    #     for i in range(10):
    #         data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
    #         dataset = TrackNetDataset(
    #             data_path,
    #             shape=shape,
    #             input_size=input_size,
    #             output_size=output_size,
    #             mode=mode
    #         )
    #         train_datasets.append(dataset)
    #     train_set = torch.utils.data.ConcatDataset(train_datasets)
    #     train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    #     test_datasets = []
    #     for i in range(10):
    #         data_path = os.path.join(dirname, f'./data/game{i+1}/Clip2')
    #         dataset = TrackNetDataset(
    #             data_path,
    #             shape=shape,
    #             input_size=input_size,
    #             output_size=output_size,
    #             mode=mode
    #         )
    #         test_datasets.append(dataset)
    #     test_set = torch.utils.data.ConcatDataset(test_datasets)
    #     test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    #     multiplier = 1 if mode == ImageReadMode.GRAY else 3
    #     model = TrackNet(multiplier*input_size, output_size, channels=channel)
    #     device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    #     model.load_state_dict(
    #         torch.load(os.path.join(dirname, f'./trained_models/tracknet/{model_name}.pt'), map_location=torch.device(device))
    #     )
    #     model.to(device)
    #     model.eval()

    #     train_statistics, train_scores = evaluate_tracknet(model, train_loader, device)
    #     test_statistics, test_scores = evaluate_tracknet(model, test_loader, device)
    #     print(model_name)
    #     print('Train Performance')
    #     print(train_statistics)
    #     print(train_scores)
    #     print('Test Performance')
    #     print(test_statistics)
    #     print(test_scores)

    """Evaluating final model."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 3
    # output_size = 3
    # mode = ImageReadMode.RGB
    # channels = [32, 64, 128, 256]
    # mode_name = 'RGB'

    # model_name = 'tracknet'

    # data_dictionary = {
    #     # 'game1': [f'Clip{i+1}' for i in range(13)],
    #     # 'game2': [f'Clip{i+1}' for i in range(8)],
    #     # 'game3': [f'Clip{i+1}' for i in range(9)],
    #     # 'game4': [f'Clip{i+1}' for i in range(7)],
    #     # 'game5': [f'Clip{i+1}' for i in range(15)],
    #     # 'game6': [f'Clip{i+1}' for i in range(4)],
    #     # 'game7': [f'Clip{i+1}' for i in range(9)],
    #     'game8': [f'Clip{i+1}' for i in range(9)],
    #     'game9': [f'Clip{i+1}' for i in range(9)],
    #     'game10': [f'Clip{i+1}' for i in range(12)],
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
    # test_set = torch.utils.data.ConcatDataset(datasets)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # model = TrackNet(3*input_size, output_size, channels=channels)
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(
    #     torch.load(os.path.join(dirname, f'./trained_models/tracknet/{model_name}.pt'), map_location=torch.device(device))
    # )
    # model.to(device)
    # model.eval()

    # test_statistics, test_scores = evaluate_tracknet(model, test_loader, device)
    # print(model_name)
    # print('Test Performance')
    # print(test_statistics)
    # print(test_scores)

