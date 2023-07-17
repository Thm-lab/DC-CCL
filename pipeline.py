import os
import time
import utils
import pickle
import datasets

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from inputimeout import inputimeout, TimeoutOccurred
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer():

    def __init__(self,
                 cfg,
                 shared_encoder=None,
                 model=None,
                 model_=None,
                 mode='co-train',
                 freeze=True,
                 criterion=nn.CrossEntropyLoss,
                 optimizer=optim.SGD,
                 scheduler=optim.lr_scheduler.StepLR,
                 T=-1,
                 criterion_=None,
                 ddp=False,
                 local_rank=0):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.shared_encoder = shared_encoder
        self.model = model
        self.model_ = model_
        self.mode = mode
        self.freeze = freeze
        self.criterion = criterion
        self.T = T
        self.criterion_ = criterion_
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ddp = ddp
        self.local_rank = local_rank
        if self.ddp == True:
            torch.cuda.set_device(self.local_rank)

    def train(self):
        # cuda
        if self.shared_encoder is not None:
            if self.ddp == True:
                self.shared_encoder = self.shared_encoder.to(self.local_rank)
                self.shared_encoder = DDP(self.shared_encoder,
                                          device_ids=[self.local_rank],
                                          output_device=self.local_rank)
            else:
                self.shared_encoder = self.shared_encoder.to(self.device)
        if self.ddp == True:
            self.model = self.model.to(self.local_rank)
            self.model = DDP(self.model,
                             device_ids=[self.local_rank],
                             output_device=self.local_rank)
        else:
            self.model = self.model.to(self.device)
        if self.model_ is not None:
            if self.ddp == True:
                self.model_ = self.model_.to(self.local_rank)
                self.model_ = DDP(self.model_,
                                  device_ids=[self.local_rank],
                                  output_device=self.local_rank)
            else:
                self.model_ = self.model_.to(self.device)
        self.criterion = self.criterion().to(self.device)
        if self.criterion_ is not None:
            self.criterion_ = self.criterion_().to(self.device)
        # Preparing data
        if 'trainset_root' in self.cfg and self.cfg['trainset_root'] != '':
            print('==> Preparing data..')
            train_mean, train_std = utils.mean_std(self.cfg['trainset_root'])
            test_mean, test_std = utils.mean_std(self.cfg['testset_root'])
            if 'train_transform' in self.cfg:
                train_trasnform = self.cfg['train_transform']
            else:
                train_trasnform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomErasing(scale=(0.04, 0.2),
                                             ratio=(0.5, 2)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.Normalize(mean=train_mean, std=train_std)
                ])
            if 'test_transform' in self.cfg:
                test_transform = self.cfg['test_transform']
            else:
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=test_mean, std=test_std),
                ])
            trainset = datasets.CIFAR10(root=self.cfg['trainset_root'],
                                        transform=train_trasnform)
            testset = datasets.CIFAR10(root=self.cfg['testset_root'],
                                       transform=test_transform)
            # Training parameters
            if all(key in self.cfg
                   for key in ['batch_size', 'shuffle', 'num_workers']):
                print('==> Loading data..')
                if self.ddp == True:
                    train_sampler = DistributedSampler(trainset)
                    trainset_loader = DataLoader(
                        trainset,
                        batch_size=self.cfg['batch_size'],
                        shuffle=self.cfg['shuffle'],
                        num_workers=self.cfg['num_workers'],
                        sampler=train_sampler)
                else:
                    trainset_loader = DataLoader(
                        trainset,
                        batch_size=self.cfg['batch_size'],
                        shuffle=self.cfg['shuffle'],
                        num_workers=self.cfg['num_workers'])
                testset_loader = DataLoader(
                    testset,
                    batch_size=self.cfg['batch_size'],
                    shuffle=self.cfg['shuffle'],
                    num_workers=self.cfg['num_workers'])
            else:
                print(
                    'cfg must contain \'batch_size\', \'shuffle\' and \'num_workers\''
                )
                return
            if all(key in self.cfg
                   for key in ['learning_rate', 'momentum', 'weight_decay']):
                print('==> Building optimizer..')
                if self.shared_encoder is not None and self.freeze is False and self.optimizer == optim.SGD:
                    shared_encoder_optimizer = self.optimizer(
                        self.shared_encoder.parameters(),
                        lr=self.cfg['learning_rate'],
                        momentum=self.cfg['momentum'],
                        weight_decay=self.cfg['weight_decay'])
                if self.optimizer == optim.SGD:
                    model_optimizer = self.optimizer(
                        self.model.parameters(),
                        lr=self.cfg['learning_rate'],
                        momentum=self.cfg['momentum'],
                        weight_decay=self.cfg['weight_decay'])
                elif self.optimizer == optim.Adam:
                    model_optimizer = self.optimizer(
                        self.model.parameters(),
                        lr=self.cfg['learning_rate'],
                        weight_decay=self.cfg['weight_decay'])

            else:
                print(
                    'cfg must contain \'learning_rate\', \'momentum\' and \'weight_decay\''
                )
                return
            if all(key in self.cfg for key in ['step_size', 'gamma']):
                print('==> Building scheduler..')
                if self.shared_encoder is not None and self.freeze is False:
                    shared_encoder_scheduler = self.scheduler(
                        optimizer=shared_encoder_optimizer,
                        step_size=self.cfg['step_size'],
                        gamma=self.cfg['gamma'],
                        last_epoch=-1)
                model_scheduler = self.scheduler(
                    optimizer=model_optimizer,
                    step_size=self.cfg['step_size'],
                    gamma=self.cfg['gamma'],
                    last_epoch=-1)
            # Model
            print('==> Building model..')
            if self.shared_encoder is not None:
                print(
                    utils.get_variable_name(self.shared_encoder) +
                    ' structure:')
                print(self.shared_encoder)
            print(utils.get_variable_name(self.model) + ' structure:')
            print(self.model)
            if self.model_ is not None:
                print(utils.get_variable_name(self.model_) + ' structure:')
                print(self.model_)
            if self.shared_encoder is not None and self.freeze is False:
                print(utils.get_variable_name(shared_encoder_optimizer) + ':')
                print(shared_encoder_optimizer)
            print(utils.get_variable_name(model_optimizer) + ' :')
            print(model_optimizer)
            if self.shared_encoder is not None and self.freeze is False:
                print(utils.get_variable_name(shared_encoder_scheduler) + ':')
                print(
                    'StepLR (\nParameter Group 0\n    step_size: {step_size}\n    gamma: {gamma}\n)'
                    .format(step_size=self.cfg['step_size'],
                            gamma=self.cfg['gamma']))
            print(utils.get_variable_name(model_scheduler) + ' :')
            print(
                'StepLR (\nParameter Group 0\n    step_size: {step_size}\n    gamma: {gamma}\n)'
                .format(step_size=self.cfg['step_size'],
                        gamma=self.cfg['gamma']))
            try:
                user_input = inputimeout(
                    prompt=
                    'Please check the model and optimizer.\nInput \'Y/y\' to continue, \'N/n\' to exit: ',
                    timeout=5)
                if user_input == 'n':
                    return
            except TimeoutOccurred:
                print('==> Continue..')
            # Training
            if 'epochs' in self.cfg:
                epochs = []
                accs = []
                iterations = []
                losses = []
                print('==> Training..')
                for i in range(self.cfg['epochs']):
                    start_time = time.time()
                    print('Epoch {}/{}'.format(i, self.cfg['epochs'] - 1))
                    if self.shared_encoder is not None:
                        self.shared_encoder.train()
                    self.model.train()
                    if self.model_ is not None:
                        self.model_.train()
                    train_loss = 0
                    correct = 0
                    total = 0
                    if self.shared_encoder is not None and self.freeze is True:
                        for name, param in self.shared_encoder.named_parameters(
                        ):
                            param.requires_grad = False
                    if self.model_ is not None and self.freeze is True:
                        for name, param in self.shared_encoder.named_parameters(
                        ):
                            param.requires_grad = False
                    for batch_idx, (x, label) in enumerate(trainset_loader):
                        x, label = x.to(self.device), label.to(self.device)
                        if self.shared_encoder is not None and self.freeze is False:
                            shared_encoder_optimizer.zero_grad()
                        model_optimizer.zero_grad()
                        if self.shared_encoder is not None:
                            x = self.shared_encoder(x)
                        y = self.model(x)
                        if self.model_ is not None and self.mode == 'co-train':
                            y_ = self.model_(x)
                            y = y.clone() + y_
                        elif self.model_ is not None and self.mode == 'control-train':
                            if self.T == -1:
                                label = self.model_(x)
                            else:
                                y_ = self.model_(x)
                                ys = F.log_softmax(y / self.T, dim=1)
                                ys_ = F.softmax(y_ / self.T, dim=1)
                                loss_ = self.criterion_(ys, ys_)
                        loss = self.criterion(y, label)
                        if self.T != -1:
                            loss = loss.clone() * (
                                1 - self.cfg['alpha']
                            ) + loss_ * self.cfg['alpha'] * self.T * self.T
                        loss.backward()
                        if self.shared_encoder is not None and self.freeze is False:
                            shared_encoder_optimizer.step()
                        model_optimizer.step()
                        train_loss += loss.item()
                        total += label.size(0)
                        if self.mode == 'co-train':
                            _, predicted = y.max(1)
                            correct += predicted.eq(label).sum().item()
                            iterations.append(batch_idx + i * len(trainset) /
                                              self.cfg['batch_size'])
                            losses.append(train_loss / (batch_idx + 1))
                            print(
                                'TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d)' %
                                (train_loss / (batch_idx + 1),
                                 100. * correct / total, correct, total))
                        elif self.mode == 'control-train':
                            iterations.append(batch_idx + i * len(trainset) /
                                              self.cfg['batch_size'])
                            losses.append(train_loss / (batch_idx + 1))
                            print('TrainLoss: %.3f | Batch: (%d)' %
                                  (train_loss / (batch_idx + 1), total))
                    if self.shared_encoder is not None and self.freeze is False:
                        shared_encoder_scheduler.step()
                    model_scheduler.step()
                    # Testing
                    if self.shared_encoder is not None:
                        self.shared_encoder.eval()
                    self.model.eval()
                    if self.model_ is not None:
                        self.model_.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch_idx, (x, label) in enumerate(testset_loader):
                            x, label = x.to(self.device), label.to(self.device)
                            if self.shared_encoder is not None:
                                x = self.shared_encoder(x)
                            y = self.model(x)
                            if self.model_ is not None and self.mode == 'co-train':
                                y_ = self.model_(x)
                                y = y.clone() + y_
                            criterion_ = nn.CrossEntropyLoss()
                            loss = criterion_(y, label)
                            test_loss += loss.item()
                            _, predicted = y.max(1)
                            total += label.size(0)
                            correct += predicted.eq(label).sum().item()
                            print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' %
                                  (test_loss / (batch_idx + 1),
                                   100. * correct / total, correct, total))

                        acc = 100. * correct / total
                        epochs.append(i)
                        accs.append(acc)
                    end_time = time.time()
                    print('Time Elapsed {:.2f} seconds'.format(end_time -
                                                               start_time))
                # Saving
                print('==> Saving result..')
                _, (ax1, ax2) = plt.subplots(2, 1)
                ax1.plot(epochs, accs)
                ax1.set_xlabel('epochs')
                ax1.set_ylabel('Accuracy')
                ax2.plot(iterations, losses)
                ax2.set_xlabel('iterations')
                ax2.set_ylabel('loss')
                if os.path.exists('./figure') is False:
                    os.mkdir('./figure')
                plt.tight_layout()
                plt.savefig('./figure/' + utils.get_variable_name(self.model) +
                            '.png')
                print('Acc and Loss saved to ./figure/' +
                      utils.get_variable_name(self.model) + '.png')
                model_result = {
                    'epochs': epochs,
                    'accs': accs,
                    'iterations': iterations,
                    'losses': losses
                }
                if os.path.exists('./result') is False:
                    os.mkdir('./result')
                with open(
                        './result/' + utils.get_variable_name(self.model) +
                        '.pkl', 'wb') as f:
                    pickle.dump(model_result, f)
                    print('Acc and Loss saved to ./result/' +
                          utils.get_variable_name(self.model) + '.pkl')
                if os.path.exists('./checkpoint') is False:
                    os.mkdir('./checkpoint')
                if self.shared_encoder is not None and self.freeze is False:
                    torch.save(
                        self.shared_encoder.state_dict(), './checkpoint/' +
                        utils.get_variable_name(self.shared_encoder) + '.pth')
                    print(
                        utils.get_variable_name(self.shared_encoder) +
                        ' parameters saved to ./checkpoint/' +
                        utils.get_variable_name(self.shared_encoder) + '.pth')
                torch.save(
                    self.model.state_dict(), './checkpoint/' +
                    utils.get_variable_name(self.model) + '.pth')
                print(
                    utils.get_variable_name(self.model) +
                    ' parameters saved to ./checkpoint/' +
                    utils.get_variable_name(self.model) + '.pth')
            else:
                print('cfg must contain \'epochs\'')
                return
        else:
            print('cfg must contain \'trainset_root\' and \'testset_root\'')
            return