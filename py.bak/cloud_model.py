import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import time
import pickle

import matplotlib.pyplot as plt

import utils
from models import Shared_encoder
from models import Cloud_model
from datasets import CIFAR10


def main():
    cfg = {
        'batch_size': 256,
        'epochs': 20,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-3
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    print('==> Preparing data..')
    train_mean, train_std = utils.mean_std('./data/cloud-samples')
    test_mean, test_std = utils.mean_std('./data/full-samples')
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

    trainset = CIFAR10(root='./data/cloud-samples', transform=transform_train)

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
    cloud_model = Cloud_model()
    cloud_model = cloud_model.to(device)

    criterion = nn.CrossEntropyLoss()

    shared_encoder_optimizer = optim.SGD(cloud_model.parameters(),
                                         lr=cfg['lr'],
                                         momentum=cfg['momentum'],
                                         weight_decay=cfg['weight_decay'])
    cloud_model_optimizer = optim.SGD(cloud_model.parameters(),
                                      lr=cfg['lr'],
                                      momentum=cfg['momentum'],
                                      weight_decay=cfg['weight_decay'])

    cloud_model_scheduler = optim.lr_scheduler.StepLR(
        optimizer=cloud_model_optimizer, step_size=5, gamma=0.4, last_epoch=-1)
    shared_encoder_scheduler = optim.lr_scheduler.StepLR(
        optimizer=shared_encoder_optimizer,
        step_size=5,
        gamma=0.4,
        last_epoch=-1)

    # Training
    def train(current, iterations, losses):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        cloud_model.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            cloud_model_optimizer.zero_grad()

            y = shared_encoder(x)
            y = cloud_model(y)
            loss = criterion(y, label)
            loss.backward()

            shared_encoder_optimizer.step()
            cloud_model_optimizer.step()

            train_loss += loss.item()
            _, predicted = y.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            iterations.append(batch_idx +
                              current * len(trainset) / cfg['batch_size'])
            losses.append(train_loss / (batch_idx + 1))
            print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' %
                  (train_loss /
                   (batch_idx + 1), 100. * correct / total, correct, total))

        shared_encoder_scheduler.step()
        cloud_model_scheduler.step()

    def test():
        cloud_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)
                y = shared_encoder(x)
                y = cloud_model(y)
                loss = criterion(y, label)

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

                print(
                    'TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' %
                    (test_loss /
                     (batch_idx + 1), 100. * correct / total, correct, total))

            acc = 100. * correct / total
            return acc

    epochs = []
    accs = []
    iterations = []
    losses = []
    for i in range(cfg['epochs']):
        start_time = time.time()
        train(i, iterations, losses)

        acc = test()
        epochs.append(i)
        accs.append(acc)

        end_time = time.time()
        print('Time Elapsed {:.2f} seconds'.format(end_time - start_time))

    _, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(epochs, accs)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Accuracy')
    # for i, txt in enumerate(accs):
    #     plt.annotate(txt, (epochs[i], accs[i]))
    ax2.plot(iterations, losses)
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('loss')

    if os.path.isdir('./figure') == False:
        os.mkdir('./figure')
    plt.tight_layout()
    plt.savefig('./figure/cloud_model.png')

    if os.path.isdir('./checkpoint') == False:
        os.mkdir('./checkpoint')
    torch.save(shared_encoder.state_dict(), './checkpoint/shared_encoder.pth')
    torch.save(cloud_model.state_dict(), './checkpoint/cloud_model.pth')

    vgg_cloud_model_result = {
        'epochs': epochs,
        'accs': accs,
        'iterations': iterations,
        'losses': losses
    }
    if os.path.isdir('./result') == False:
        os.mkdir('./result')
    with open('./result/cloud_model.pkl', 'wb') as f:
        pickle.dump(vgg_cloud_model_result, f)


if __name__ == '__main__':
    main()