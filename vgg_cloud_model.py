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
from models import VGG_shared_encoder
from models import VGG_cloud_model
from datasets import CIFAR10


def main():
    cfg = {
        'batch_size': 256,
        'epochs': 30,
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
    vgg_shared_encoder = VGG_shared_encoder()
    vgg_shared_encoder = vgg_shared_encoder.to(device)
    vgg_cloud_model = VGG_cloud_model()
    vgg_cloud_model = vgg_cloud_model.to(device)

    criterion = nn.CrossEntropyLoss()

    vgg_shared_encoder_optimizer = optim.SGD(vgg_cloud_model.parameters(),
                                             lr=cfg['lr'],
                                             momentum=cfg['momentum'],
                                             weight_decay=cfg['weight_decay'])
    vgg_cloud_model_optimizer = optim.SGD(vgg_cloud_model.parameters(),
                                          lr=cfg['lr'],
                                          momentum=cfg['momentum'],
                                          weight_decay=cfg['weight_decay'])

    vgg_cloud_model_scheduler = optim.lr_scheduler.StepLR(
        optimizer=vgg_cloud_model_optimizer,
        step_size=5,
        gamma=0.4,
        last_epoch=-1)
    vgg_shared_encoder_scheduler = optim.lr_scheduler.StepLR(
        optimizer=vgg_shared_encoder_optimizer,
        step_size=5,
        gamma=0.4,
        last_epoch=-1)

    # Training
    def train(current, iterations, losses):
        print('Epoch {}/{}'.format(current, cfg['epochs'] - 1))
        vgg_cloud_model.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            vgg_cloud_model_optimizer.zero_grad()

            y = vgg_shared_encoder(x)
            y = vgg_cloud_model(y)
            loss = criterion(y, label)
            loss.backward()

            vgg_shared_encoder_optimizer.step()
            vgg_cloud_model_optimizer.step()

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

        vgg_shared_encoder_scheduler.step()
        vgg_cloud_model_scheduler.step()

    def test():
        vgg_cloud_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)
                y = vgg_shared_encoder(x)
                y = vgg_cloud_model(y)
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
    plt.savefig('./figure/vgg_cloud_model.png')

    if os.path.isdir('./checkpoint') == False:
        os.mkdir('./checkpoint')
    torch.save(vgg_shared_encoder.state_dict(),
               './checkpoint/vgg_shared_encoder.pth')
    torch.save(vgg_cloud_model.state_dict(),
               './checkpoint/vgg_cloud_model.pth')

    vgg_cloud_model_result = {
        'epochs': epochs,
        'accs': accs,
        'iterations': iterations,
        'losses': losses
    }
    if os.path.isdir('./result') == False:
        os.mkdir('./result')
    with open('./result/vgg_cloud_model.pkl', 'wb') as f:
        pickle.dump(vgg_cloud_model_result, f)


if __name__ == '__main__':
    main()