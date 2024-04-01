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

from dataloaders.datasets import TrackNetDataset
from models.tracknet import TrackNet
from losses.focal_loss import BinaryFocalLoss
from metrics.evaluator import evaluate_tracknet


# TODO: Move training to a trainer class
if __name__ == '__main__':
    """Training tuning models."""
    # dirname = os.path.dirname(__file__)
    # shape = (288, 512)
    # input_size = 5
    # output_sizes = (3, 5)
    # modes = (ImageReadMode.GRAY, ImageReadMode.RGB)
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
    # dirname = os.path.dirname(__file__)

    # input_size = 3
    # output_size = 1
    # shape = (288, 512)

    # model = TrackNet(input_size, output_size)
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(torch.load(os.path.join(dirname, './weights/small_tracknet.pt'), map_location=torch.device(device)))
    # model.to(device)

    # path = os.path.join(dirname, './data/game1/Clip1')
    # dataset = TrackNetDataset(
    #     path,
    #     shape=shape,
    #     input_size=input_size,
    #     output_size=output_size,
    #     mode=ImageReadMode.GRAY
    # )

    # image_names = pd.read_csv(f'{path}/Label.csv')['file name'].tolist()[2:]

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(os.path.join(dirname, './videos/test.mp4'), fourcc, 30, (1280, 720))

    # model.eval()
    # for name, input_ in zip(image_names, dataset):
    #     pred = model(input_[0].unsqueeze(0))
    #     ball = model.detect_ball(pred)
    #     image = cv2.imread(f'{path}/{name}')
    #     if ball[0] == -1 or ball[1] == -1:
    #         video.write(image)
    #     else:
    #         ball = (int(2.5*ball[0]), int(2.5*ball[1]))
    #         image = cv2.circle(image, ball, 5, (0, 0, 255), -1)
    #         video.write(image)
    #     cv2.imshow('frame', image)
    #     cv2.waitKey(1000)
    # video.release()
    # cv2.destroyAllWindows()
    # exit()

    """Evaluating tuning models."""
    dirname = os.path.dirname(__file__)
    shape = (288, 512)
    input_size = 3
    output_sizes = (3,)
    modes = (ImageReadMode.RGB,)

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
            model.load_state_dict(torch.load(os.path.join(dirname, f'./weights/{model_name}.pt'), map_location=torch.device(device)))
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