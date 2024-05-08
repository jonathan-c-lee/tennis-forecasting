"""Main script."""
import os
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

from models.tracknet import TrackNet
from models.trajectory_predictor import *


if __name__ == '__main__':
    """Creating video."""
    dirname = os.path.dirname(__file__)

    # load tracknet
    model_name = 'tracknet'
    config = None
    with open(os.path.join(dirname, f'./configs/tracknet/{model_name}.yaml'), 'r') as file:
        config = yaml.full_load(file)
    input_size, output_size = config['frames_in'], config['frames_out']
    shape = config['shape_in']
    mode = ImageReadMode.RGB if config['mode'] == 'RGB' else ImageReadMode.GRAY
    multiplier = 3 if config['mode'] == 'RGB' else 1
    channels = config['channels']
    model = TrackNet(multiplier*input_size, output_size, channels=channels)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(os.path.join(dirname, f'./trained_models/tracknet/{model_name}.pt'), map_location=torch.device(device))
    )
    model.to(device)

    # video setup
    raw_video = cv2.VideoCapture(os.path.join(dirname, f'./videos/test/test.mp4'))
    video_in = cv2.VideoCapture(os.path.join(dirname, f'./videos/test/test_inter.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(os.path.join(dirname, './videos/test/test_out.mp4'), fourcc, 30, (1280, 720))

    # define image transform
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(shape)
    ])

    # get ball position
    model.eval()
    ball_positions = []
    in_images = []
    has_next = True
    while has_next:
        images = []
        for i in range(3):
            success, raw_image = raw_video.read()
            if not success:
                has_next = False
                break
            _, in_image = video_in.read()
            in_images.append(in_image)
            images.append(np.moveaxis(raw_image, -1, 0))
        if not has_next:
            break
        input_ = torch.cat([transform(torch.from_numpy(image)) for image in images], dim=0).to(device)

        predictions = model(input_.unsqueeze(0)).squeeze()
        for heatmap in predictions:
            ball = model.detect_ball(heatmap)
            if ball[0] == -1 or ball[1] == -1:
                ball_positions.append(ball)
            else:
                ball = (int(2.5*ball[0]), int(2.5*ball[1]))
                ball_positions.append(ball)

    # for i in tqdm(range(len(ball_positions))):
    #     image = raw_images[i]
    #     ball = ball_positions[i]
    #     if ball[0] < 0 or ball[1] < 0:
    #         video_out.write(image)
    #         continue
    #     image = cv2.circle(image, ball, 4, (0, 0, 255), -1)
    #     video_out.write(image)
    # video_out.release()
    # cv2.destroyAllWindows()
    # exit()

    # load trajectory predictor
    model_name = 'position_trajectory_predictor'
    config = None
    with open(os.path.join(dirname, f'./configs/position_trajectory/{model_name}.yaml'), 'r') as file:
        config = yaml.full_load(file)
    input_frames, output_frames = config['frames_in'], config['frames_out']
    position_dim = config['position_dim']
    hidden_dim = config['hidden_size']
    lstm_layers = config['layers']
    dropout = config['dropout']
    model = PositionTrajectoryPredictor(output_frames, position_dim, hidden_dim, lstm_layers, dropout)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(os.path.join(dirname, f'./trained_models/position_trajectory/{model_name}.pt'), map_location=torch.device(device))
    )
    model.to(device)
    model.eval()

    player_positions = pd.read_csv(os.path.join(dirname, f'./boxes_test_video.csv'))
    p1x = (player_positions['p1_x1'] + player_positions['p1_x2']) / 2
    p1y = (player_positions['p1_y1'] + player_positions['p1_y2']) / 2
    p2x = (player_positions['p2_x1'] + player_positions['p2_x2']) / 2
    p2y = (player_positions['p2_y1'] + player_positions['p2_y2']) / 2
    p1x /= 1280
    p1y /= 720
    p2x /= 1280
    p2y /= 720
    p1x = p1x.to_list()
    p1y = p1y.to_list()
    p2x = p2x.to_list()
    p2y = p2y.to_list()
    positions = [position for position in zip(p1x, p1y, p2x, p2y)]

    normal_ball = [(float(ball[0]) / 1280, float(ball[1]) / 720) for ball in ball_positions]
    predictions = []
    limit = min(len(positions), len(normal_ball))
    positions = positions[:limit]
    normal_ball = normal_ball[:limit]
    for i in tqdm(range(len(normal_ball))):
        image = in_images[i]
        b_inputs = normal_ball[max(0, i+1-input_frames):i+1]
        b_inputs = [ball for ball in b_inputs if ball[0] >= 0 and ball[1] >= 0]
        p_inputs = positions[max(0, i+1-input_frames):i+1]
        if len(b_inputs) != input_frames or len(p_inputs) != input_frames:
            video_out.write(image)
            continue
        
        ball_inputs = torch.stack([torch.tensor(b) for b in b_inputs], dim=0).to(device)
        ball_inputs = torch.unsqueeze(ball_inputs, 0).to(device)
        position_inputs = torch.stack([torch.tensor(p) for p in p_inputs], dim=0).to(device)
        position_inputs = torch.unsqueeze(position_inputs, 0).to(device)
        outputs = torch.squeeze(model(ball_inputs, position_inputs), 0)
        print(outputs)
        for ball in torch.squeeze(ball_inputs, 0):
            ball = (int(1280*ball[0]), int(720*ball[1]))
            image = cv2.circle(image, ball, 4, (0, 0, 255), -1)
        if ball_positions[i][0] != -1 and ball_positions[i][1] != -1:
            image = cv2.circle(image, ball_positions[i], 4, (255, 0, 255), -1)
        for j in range(0, len(outputs), 2):
            print('painting output')
            ball = (int(1280*outputs[j]), int(720*outputs[j+1]))
            image = cv2.circle(image, ball, 4, (255, 0, 0), -1)
        reals = normal_ball[i+1:min(len(normal_ball), i+1+output_frames)]
        for ball in reals:
            if ball[0] >= 0 and ball[1] >= 0:
                ball = (int(1280*ball[0]), int(720*ball[1]))
                image = cv2.circle(image, ball, 4, (0, 255, 0), -1)
        video_out.write(image)
    video_out.release()
    cv2.destroyAllWindows()
    exit()

