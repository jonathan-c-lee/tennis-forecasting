"""Main script."""
import os
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

from dataloaders.datasets import TrackNetDataset
from models.tracknet import TrackNet
from losses.focal_loss import BinaryFocalLoss
from metrics.evaluator import evaluate_tracknet


# TODO: Move training to a trainer class
if __name__ == '__main__':
    """Training final model."""
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
    #     'weights_file': f'../weights/{model_name}.pt',
    # }
    # print(config)

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

    # model.train()
    # train_loss = []
    # for epoch in tqdm(range(epochs)):
    #     running_loss = 0.0
    #     count = 0
    #     for inputs, labels in dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = loss_criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()
    #         count += 1
    #     train_loss.append(running_loss / count)

    #     torch.save(model.state_dict(), os.path.join(dirname, f'./weights/{model_name}_{epoch+1}.pt'))
    #     with open(os.path.join(dirname, f'./configs/{model_name}_{epoch+1}.yaml'), 'w') as outfile:
    #         config['epochs'] = epoch+1
    #         config['train loss'] = train_loss
    #         yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training different sized models."""
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
    #         'weights_file': f'../weights/{model_name}.pt',
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

    #     model.train()
    #     train_loss = []
    #     for epoch in tqdm(range(epochs)):
    #         running_loss = 0.0
    #         count = 0
    #         for inputs, labels in dataloader:
    #             inputs, labels = inputs.to(device), labels.to(device)

    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = loss_criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()
    #             count += 1
    #         train_loss.append(running_loss / count)

    #     torch.save(model.state_dict(), os.path.join(dirname, f'./weights/{model_name}.pt'))
    #     with open(os.path.join(dirname, f'./configs/{model_name}.yaml'), 'w') as outfile:
    #         config['train loss'] = train_loss
    #         yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Training tuning models."""
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
    #             'weights_file': f'../weights/{model_name}.pt',
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

    #         model.train()
    #         train_loss = []
    #         for epoch in tqdm(range(epochs)):
    #             running_loss = 0.0
    #             count = 0
    #             for inputs, labels in dataloader:
    #                 inputs, labels = inputs.to(device), labels.to(device)

    #                 optimizer.zero_grad()
    #                 outputs = model(inputs)
    #                 loss = loss_criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()

    #                 running_loss += loss.item()
    #                 count += 1
    #             train_loss.append(running_loss / count)

    #         torch.save(model.state_dict(), os.path.join(dirname, f'./weights/{model_name}.pt'))
    #         with open(os.path.join(dirname, f'./configs/{model_name}.yaml'), 'w') as outfile:
    #             config['train loss'] = train_loss
    #             yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    """Creating video."""
    dirname = os.path.dirname(__file__)
    model_name = 'tracknet'
    config = None
    with open(os.path.join(dirname, f'./configs/{model_name}.yaml'), 'r') as file:
        config = yaml.full_load(file)
    
    input_size = config['frames_in']
    output_size = config['frames_out']
    shape = config['shape_in']
    mode = ImageReadMode.RGB if config['mode'] == 'RGB' else ImageReadMode.GRAY
    multiplier = 3 if config['mode'] == 'RGB' else 1
    channels = config['channels']

    model = TrackNet(multiplier*input_size, output_size, channels=channels)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(dirname, f'./weights/{model_name}.pt'), map_location=torch.device(device)))
    model.to(device)

    video_in = cv2.VideoCapture(os.path.join(dirname, f'./videos/test/test.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(os.path.join(dirname, './videos/test/test_out.mp4'), fourcc, 30, (1280, 720))

    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(shape)
    ])

    model.eval()
    has_next = True
    count = 0
    while has_next:
        print(count)
        images = []
        for _ in range(3):
            success, image = video_in.read()
            if not success:
                has_next = False
                break
            images.append(image)
        if not has_next:
            break
        input_ = torch.cat([transform(torch.from_numpy(image)) for image in images], dim=0)

        predictions = model(input_.unsqueeze(0))
        for raw_image, heatmap in zip(images, predictions):
            ball = model.detect_ball(heatmap)
            if ball[0] == -1 or ball[1] == -1:
                video_out.write(raw_image)
            else:
                ball = (int(2.5*ball[0]), int(2.5*ball[1]))
                image = cv2.circle(raw_image, ball, 4, (0, 0, 255), -1)
                video_out.write(image)
        count += 1
    video_out.release()
    cv2.destroyAllWindows()
    exit()

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
    #         model.load_state_dict(torch.load(os.path.join(dirname, f'./weights/{model_name}.pt'), map_location=torch.device(device)))
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
    #     model.load_state_dict(torch.load(os.path.join(dirname, f'./weights/{model_name}.pt'), map_location=torch.device(device)))
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
    # model.load_state_dict(torch.load(os.path.join(dirname, f'./weights/{model_name}.pt'), map_location=torch.device(device)))
    # model.to(device)
    # model.eval()

    # test_statistics, test_scores = evaluate_tracknet(model, test_loader, device)
    # print(model_name)
    # print('Test Performance')
    # print(test_statistics)
    # print(test_scores)