import os
import time
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import utils
from models import VGG_shared_encoder
from models import VGG_cloud_model
from models import VGG_control_model
from datasets import CIFAR10


def main():
    cfg = {'batch_size': 256, 'epochs': 15, 'lr': 0.001, 'weight_decay': 0}

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
    vgg_shared_encoder = VGG_shared_encoder()
    vgg_shared_encoder = vgg_shared_encoder.to(device)
    vgg_shared_encoder.load_state_dict(
        torch.load('./checkpoint/vgg_shared_encoder.pth'))
    vgg_cloud_model = VGG_cloud_model()
    vgg_cloud_model = vgg_cloud_model.to(device)
    vgg_cloud_model.load_state_dict(
        torch.load('./checkpoint/vgg_cloud_model.pth'))
    vgg_control_model = VGG_control_model()
    vgg_control_model = vgg_control_model.to(device)

    criterion = nn.MSELoss()
    vgg_control_model_optimizer = optim.Adam(vgg_control_model.parameters(),
                                             lr=cfg['lr'],
                                             weight_decay=cfg['weight_decay'])

    vgg_control_model_scheduler = optim.lr_scheduler.StepLR(
        optimizer=vgg_control_model_optimizer,
        step_size=10,
        gamma=0.1,
        last_epoch=-1)

    # Training
    def train(current, iterations, losses):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        vgg_cloud_model.train()
        vgg_control_model.train()
        for name, param in vgg_shared_encoder.named_parameters():
            param.requires_grad = False
        for name, param in vgg_cloud_model.named_parameters():
            param.requires_grad = False

        train_loss = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            vgg_control_model_optimizer.zero_grad()

            y = vgg_shared_encoder(x)
            label = vgg_cloud_model(y)
            y = vgg_control_model(y)
            loss = criterion(y, label)
            loss.backward()

            vgg_control_model_optimizer.step()

            train_loss += loss.item()

            iterations.append(batch_idx +
                              current * len(trainset) / cfg['batch_size'])
            losses.append(train_loss / (batch_idx + 1))

            print('TrainLoss: %.3f' % (train_loss / (batch_idx + 1)))

        vgg_control_model_scheduler.step()

    iterations = []
    losses = []
    for i in range(cfg['epochs']):
        start_time = time.time()
        train(i, iterations, losses)
        end_time = time.time()
        print('Time Elapsed {:.2f} seconds'.format(end_time - start_time))

    plt.plot(iterations, losses)
    plt.xlabel('iterations')
    plt.ylabel('loss')

    if os.path.isdir('./figure') == False:
        os.mkdir('./figure')
    plt.tight_layout()
    plt.savefig('./figure/vgg_control_model.png')

    torch.save(vgg_control_model.state_dict(),
               './checkpoint/vgg_control_model.pth')


if __name__ == '__main__':
    main()