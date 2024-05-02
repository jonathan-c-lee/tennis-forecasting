"""Main script."""
import os
import yaml
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

from models.tracknet import TrackNet
from models.trajectory_predictor import TrajectoryBaseline


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
    video_in = cv2.VideoCapture(os.path.join(dirname, f'./videos/test/test.mp4'))
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
    raw_images = []
    has_next = True
    count = 0
    while has_next:
        print(count)
        count += 1
        images = []
        for _ in range(3):
            success, image = video_in.read()
            if not success:
                has_next = False
                break
            raw_images.append(image)
            mod_image = np.moveaxis(image, -1, 0)
            images.append(mod_image)
        if not has_next:
            break
        input_ = torch.cat([transform(torch.from_numpy(image)) for image in images], dim=0).to(device)

        predictions = model(input_.unsqueeze(0)).squeeze()
        for raw_image, heatmap in zip(images, predictions):
            ball = model.detect_ball(heatmap)
            if ball[0] == -1 or ball[1] == -1:
                ball_positions.append(ball)
            else:
                ball = (int(2.5*ball[0]), int(2.5*ball[1]))
                ball_positions.append(ball)

    # load trajectory predictor
    model_name = 'baseline_trajectory_predictor'
    config = None
    with open(os.path.join(dirname, f'./configs/trajectory/{model_name}.yaml'), 'r') as file:
        config = yaml.full_load(file)
    input_frames, output_frames = config['frames_in'], config['frames_out']
    hidden_dim = config['hidden_size']
    lstm_layers = config['layers']
    dropout = config['dropout']
    model = TrajectoryBaseline(output_frames, hidden_dim, lstm_layers, dropout)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(os.path.join(dirname, f'./trained_models/trajectory/{model_name}.pt'), map_location=torch.device(device))
    )
    model.to(device)
    model.eval()

    normal_ball = [(float(ball[0]) / 1280, float(ball[1]) / 720) for ball in ball_positions]
    predictions = []
    for i in tqdm(range(len(normal_ball))):
        image = raw_images[i]
        inputs = normal_ball[max(0, i+1-input_frames):i+1]
        inputs = [ball for ball in inputs if ball[0] >= 0 and ball[1] >= 0]
        if len(inputs) == 0 or len(inputs) != input_frames:
            video_out.write(image)
            continue
        inputs = torch.stack([torch.tensor(ball) for ball in inputs], dim=0).to(device)
        inputs = torch.unsqueeze(inputs, 0).to(device)
        outputs = torch.squeeze(model(inputs), 0)
        for ball in torch.squeeze(inputs, 0):
            ball = (int(1280*ball[0]), int(720*ball[1]))
            image = cv2.circle(image, ball, 4, (0, 0, 255), -1)
        if ball_positions[i][0] != -1 and ball_positions[i][1] != -1:
            image = cv2.circle(image, ball_positions[i], 4, (255, 0, 255), -1)
        for j in range(0, len(outputs), 2):
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

