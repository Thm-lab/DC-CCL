import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import time
import argparse

import utils
from models import VGG_shared_encoder
from models import VGG_co_submodel
from models import VGG_control_model
from datasets import CIFAR10


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host',
                        required=True,
                        help='host must be : device or cloud')
    args = parser.parse_args()
    if args.host == 'device':
        traindata_root = './data/device-enhanced-samples'
    elif args.host == 'cloud':
        traindata_root = './data/cloud-samples'
    else:
        print('host must be : device or cloud')
        exit(0)

    cfg = {
        'batch_size': 256,
        'epochs':
        4 if args.host == 'device' else 1 if args.host == 'cloud' else 0,
        'lr': 0.01,
        'weight_decay': 0
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')

    train_mean, train_std = utils.mean_std(traindata_root)
    test_mean = train_mean
    test_std = train_std
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=test_mean, std=test_std),
    ])

    trainset = CIFAR10(root=traindata_root, transform=transform_train)

    trainloader = DataLoader(trainset,
                             batch_size=cfg['batch_size'],
                             shuffle=True,
                             num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=cfg['batch_size'],
                            shuffle=False,
                            num_workers=2)

    # Model
    print('==> Building model..')
    vgg_shared_encoder = VGG_shared_encoder()
    vgg_shared_encoder = vgg_shared_encoder.to(device)
    vgg_shared_encoder.load_state_dict(
        torch.load('./checkpoint/vgg_shared_encoder.pth'))
    vgg_co_control_model = VGG_co_submodel()
    vgg_co_control_model = vgg_co_control_model.to(device)
    if os.path.isfile('./checkpoint/vgg_co_control_model.pth'):
        vgg_co_control_model.load_state_dict(
            torch.load('./checkpoint/vgg_co_control_model.pth'))
    vgg_control_model = VGG_control_model()
    vgg_control_model = vgg_control_model.to(device)
    vgg_control_model.load_state_dict(
        torch.load('./checkpoint/vgg_control_model.pth'))

    criterion = nn.CrossEntropyLoss()

    vgg_co_control_model_optimizer = optim.SGD(
        vgg_co_control_model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'])

    # Training
    def train(current, iterations, accs):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        vgg_co_control_model.train()
        for name, param in vgg_shared_encoder.named_parameters():
            param.requires_grad = False

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            vgg_co_control_model_optimizer.zero_grad()

            x = vgg_shared_encoder(x)
            y = vgg_co_control_model(x)
            y_ = vgg_co_control_model(x)
            # 用clone来进行out-place修改，否则会出现in-place operation的错误
            y = y.clone() + y_

            loss = criterion(y, label)
            loss.backward()

            vgg_co_control_model_optimizer.step()

            train_loss += loss.item()
            _, predicted = y.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' %
                  (train_loss /
                   (batch_idx + 1), 100. * correct / total, correct, total))

    def test():
        vgg_shared_encoder.eval()
        vgg_co_control_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)
                x = vgg_shared_encoder(x)
                y = vgg_co_control_model(x)
                y_ = vgg_control_model(x)
                y += y_

                loss = criterion(y, label)

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            acc = 100. * correct / total
            return acc

    epochs = []
    accs = []
    for i in range(cfg['epochs']):
        start_time = time.time()
        train(i, epochs, accs)

        acc = test()
        epochs.append(i)
        accs.append(acc)

        end_time = time.time()
        print('Time Elapsed {:.2f} seconds'.format(end_time - start_time))

    print(accs)

    torch.save(vgg_co_control_model.state_dict(),
               './checkpoint/vgg_co_control_model.pth')


if __name__ == '__main__':
    main()