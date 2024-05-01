"""Main script."""
import os
import yaml
import cv2
import torch
from torchvision.io import ImageReadMode
from torchvision.transforms import v2

from models.tracknet import TrackNet


if __name__ == '__main__':
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

