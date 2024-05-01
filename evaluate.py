"""Evaluation script."""
import os
import torch
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode

from dataloaders.datasets import TrackNetDataset
from models.tracknet import TrackNet
from metrics.evaluator import evaluate_tracknet


if __name__ == '__main__':
    """Evaluating tuning models."""
    dirname = os.path.dirname(__file__)
    shape = (288, 512)
    input_size = 5
    output_sizes = (3, 5)
    modes = (ImageReadMode.GRAY, ImageReadMode.RGB)

    for output_size in output_sizes:
        for mode in modes:
            mode_name = 'GRAY' if mode == ImageReadMode.GRAY else 'RGB'
            model_name = f'tracknet_tuning_{input_size}in_{output_size}out_{mode_name}'

            train_datasets = []
            for i in range(10):
                data_path = os.path.join(dirname, f'./data/game{i+1}/Clip1')
                dataset = TrackNetDataset(
                    data_path,
                    shape=shape,
                    input_size=input_size,
                    output_size=output_size,
                    mode=mode
                )
                train_datasets.append(dataset)
            train_set = torch.utils.data.ConcatDataset(train_datasets)
            train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

            test_datasets = []
            for i in range(10):
                data_path = os.path.join(dirname, f'./data/game{i+1}/Clip2')
                dataset = TrackNetDataset(
                    data_path,
                    shape=shape,
                    input_size=input_size,
                    output_size=output_size,
                    mode=mode
                )
                test_datasets.append(dataset)
            test_set = torch.utils.data.ConcatDataset(test_datasets)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

            multiplier = 1 if mode == ImageReadMode.GRAY else 3
            model = TrackNet(multiplier*input_size, output_size)
            device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(
                torch.load(os.path.join(dirname, f'./trained_models/tracknet/{model_name}.pt'), map_location=torch.device(device))
            )
            model.to(device)
            model.eval()

            train_statistics, train_scores = evaluate_tracknet(model, train_loader, device)
            test_statistics, test_scores = evaluate_tracknet(model, test_loader, device)
            print(model_name)
            print('Train Performance')
            print(train_statistics)
            print(train_scores)
            print('Test Performance')
            print(test_statistics)
            print(test_scores)

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

