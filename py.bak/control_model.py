import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import utils
from models import Shared_encoder
from models import Cloud_model
from models import Control_model
from datasets import CIFAR10


def main():
    cfg = {'batch_size': 128, 'epochs': 15, 'lr': 0.001, 'weight_decay': 0}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    train_mean, train_std = utils.mean_std('./data/cloud-samples')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    trainset = CIFAR10(root='./data/cloud-samples', transform=transform_train)

    trainloader = DataLoader(trainset,
                             batch_size=cfg['batch_size'],
                             shuffle=True,
                             num_workers=2)

    # Model
    print('==> Building model..')
    shared_encoder = Shared_encoder()
    shared_encoder = shared_encoder.to(device)
    shared_encoder.load_state_dict(
        torch.load('./checkpoint/shared_encoder.pth'))
    cloud_model = Cloud_model()
    cloud_model = cloud_model.to(device)
    cloud_model.load_state_dict(torch.load('./checkpoint/cloud_model.pth'))
    control_model = Control_model()
    control_model = control_model.to(device)

    criterion = nn.MSELoss()
    control_model_optimizer = optim.Adam(control_model.parameters(),
                                         lr=cfg['lr'],
                                         weight_decay=cfg['weight_decay'])

    # Training
    def train(current):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        cloud_model.train()
        control_model.train()
        for name, param in shared_encoder.named_parameters():
            param.requires_grad = False
        for name, param in cloud_model.named_parameters():
            param.requires_grad = False

        train_loss = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            control_model_optimizer.zero_grad()

            y = shared_encoder(x)
            label = cloud_model(y)
            y = control_model(y)
            loss = criterion(y, label)
            loss.backward()

            control_model_optimizer.step()

            train_loss += loss.item()
            print('TrainLoss: %.3f' % (train_loss / (batch_idx + 1)))

    for i in range(cfg['epochs']):
        start_time = time.time()
        train(i)
        end_time = time.time()
        print('Time Elapsed {:.2f} seconds'.format(end_time - start_time))

    torch.save(control_model.state_dict(), './checkpoint/control_model.pth')


if __name__ == '__main__':
    main()