"""Main script."""
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.io import ImageReadMode

from dataloaders.datasets import TrackNetDataset
from models.tracknet import TrackNet
from losses.focal_loss import BinaryFocalLoss


# TODO: Move training to a trainer class
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)

    input_size = 3
    output_size = 1

    model = TrackNet(input_size, output_size)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # model.load_state_dict(torch.load(os.path.join(dirname, './weights/small_tracknet.pt'), map_location=torch.device(device)))
    model.to(device)

    path = os.path.join(dirname, './data/game1/Clip1')
    dataset = TrackNetDataset(
        path,
        shape=(288, 512),
        input_size=input_size,
        output_size=output_size,
        mode=ImageReadMode.GRAY
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    loss_criterion = BinaryFocalLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    model.train()
    train_loss = []
    epochs = 50
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_loss.append(running_loss)
        print(f'Epoch {epoch} Loss:', running_loss)
    
    print(train_loss)

    torch.save(model.state_dict(), os.path.join(dirname, './weights/small_tracknet.pt'))

    # model.eval()
    # # pred = model(dataset[0][0])
    # print(np.where(dataset[0][1].numpy() == 1))
    # print(model.detect_ball(dataset[0][1]))