"""Main script."""
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from losses.focal_loss import BinaryFocalLoss
from torchvision.io import ImageReadMode

from dataloaders.datasets import TrackNetDataset
from models.tracknet import TrackNet


# TODO: Move training to a trainer class
if __name__ == '__main__':
    input_size = 3
    output_size = 1

    model = TrackNet(input_size, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, './data/game1/Clip1')
    dataset = TrackNetDataset(
        path,
        shape=(288, 512),
        input_size=input_size,
        output_size=output_size,
        mode=ImageReadMode.GRAY
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

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