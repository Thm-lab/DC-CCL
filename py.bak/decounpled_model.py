import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import time

import utils
from models import Shared_encoder
from models import Co_submodel
from models import Cloud_model


def main():
    cfg = {'batch_size': 128, 'epochs': 1, 'lr': 0.01, 'weight_decay': 0}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    train_mean, train_std = utils.mean_std('./data/full-samples')
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

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

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
    shared_encoder = Shared_encoder()
    shared_encoder = shared_encoder.to(device)
    shared_encoder.load_state_dict(
        torch.load('./checkpoint/shared_encoder.pth'))
    cloud_model = Cloud_model()
    cloud_model = cloud_model.to(device)
    cloud_model.load_state_dict(torch.load('./checkpoint/cloud_model.pth'))
    decounpled_model = Co_submodel()
    decounpled_model = decounpled_model.to(device)

    criterion = nn.CrossEntropyLoss()

    decounpled_model_optimizer = optim.SGD(decounpled_model.parameters(),
                                           lr=cfg['lr'],
                                           weight_decay=cfg['weight_decay'])

    # Training
    def train(current, iterations, accs):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        decounpled_model.train()
        for name, param in shared_encoder.named_parameters():
            param.requires_grad = False
        for name, param in cloud_model.named_parameters():
            param.requires_grad = False

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            decounpled_model_optimizer.zero_grad()

            x = shared_encoder(x)
            y = decounpled_model(x)
            y_ = cloud_model(x)
            y += y_
            loss = criterion(y, label)
            loss.backward()

            decounpled_model_optimizer.step()

            train_loss += loss.item()
            _, predicted = y.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' %
                  (train_loss /
                   (batch_idx + 1), 100. * correct / total, correct, total))

    def test():
        decounpled_model.eval()
        cloud_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)
                x = shared_encoder(x)
                y = decounpled_model(x)
                y_ = cloud_model(x)
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

    torch.save(decounpled_model.state_dict(),
               './checkpoint/decounpled_model.pth')


if __name__ == '__main__':
    main()