import math
import random
import shutil
import cv2
import os
import json
import pickle
import inspect
import re
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import CIFAR10


def download():
    cifar10 = torchvision.datasets.CIFAR10(root='data',
                                           train=True,
                                           download=True)
    cifar10_test = torchvision.datasets.CIFAR10(root='data',
                                                train=False,
                                                download=True)
    print(cifar10)
    print(cifar10_test)


def extract():
    train_filenames = []
    train_annotations = []
    test_filenames = []
    test_annotations = []

    anno_path = './data/annotations/'
    train_path = './data/train/'
    test_path = './data/test/'
    file_dir = './data/cifar-10-batches-py'

    if os.path.exists(train_path) == False:
        os.mkdir(train_path)
    if os.path.exists(test_path) == False:
        os.mkdir(test_path)
    if os.path.exists(anno_path) == False:
        os.mkdir(anno_path)

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def cifar10_img(file_dir):
        for i in range(1, 6):
            data_name = file_dir + '/' + 'data_batch_' + str(i)
            data_dict = unpickle(data_name)
            print(data_name + ' is processing')

            for j in range(10000):
                img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_name = train_path + str(
                    data_dict[b'labels'][j]) + str((i) * 10000 + j) + '.jpg'
                annot_img_name = str(
                    data_dict[b'labels'][j]) + str((i) * 10000 + j) + '.jpg'
                img_annotations = data_dict[b'labels'][j]
                train_filenames.append(annot_img_name)
                train_annotations.append(img_annotations)

                cv2.imwrite(img_name, img)

            print(data_name + ' is done')

        test_data_name = file_dir + '/test_batch'
        print(test_data_name + ' is processing')
        test_dict = unpickle(test_data_name)

        for m in range(10000):
            img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_name = test_path + str(
                test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
            annot_img_name = str(
                test_dict[b'labels'][m]) + str(10000 + m) + '.jpg'
            img_annotations = test_dict[b'labels'][m]
            test_filenames.append(annot_img_name)
            test_annotations.append(img_annotations)
            cv2.imwrite(img_name, img)
        print(test_data_name + ' is done')
        print('Finish transforming to image')

    cifar10_img(file_dir)

    train_annot_dict = {'images': train_filenames, 'labels': train_annotations}
    test_annot_dict = {'images': test_filenames, 'labels': test_annotations}

    train_json = json.dumps(train_annot_dict)
    train_file = open('./data/annotations/train.json', 'w')
    train_file.write(train_json)
    train_file.close()

    test_json = json.dumps(test_annot_dict)
    test_file = open('./data/annotations/test.json', 'w')
    test_file.write(test_json)
    test_file.close()
    print('annotations have writen to json file')


def spilit():
    classes = [
        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
        'truck'
    ]
    ratio = 8

    with open('./data/annotations/train.json') as f:
        data = json.load(f)

    images = data['images']
    labels = data['labels']

    samples = {}
    for key in classes:
        samples[key] = []

    for image, label in zip(images, labels):
        samples[classes[label]].append(image)
        samples[classes[label]]

    classes_root = r'./data/full-samples'
    if os.path.exists(classes_root) == False:
        os.mkdir(classes_root)

    for key in classes:
        src_path = os.path.join('./data/train')
        dst_path = os.path.join(classes_root + '/', key)
        if os.path.exists(dst_path) == False:
            os.mkdir(dst_path)
        for file_name in samples[key]:
            src_file = os.path.join(src_path, file_name)
            dst_file = os.path.join(dst_path, file_name)
            shutil.copy(src_file, dst_file)
            print('Copying ' + src_file + ' to ' + dst_file)

    os.rename('./data/test', './data/tmp')
    with open('./data/annotations/test.json') as f:
        data = json.load(f)

    images = data['images']
    labels = data['labels']

    samples = {}
    for key in classes:
        samples[key] = []

    for image, label in zip(images, labels):
        samples[classes[label]].append(image)
        samples[classes[label]]

    classes_root = r'./data/test'
    if os.path.exists(classes_root) == False:
        os.mkdir(classes_root)

    for key in classes:
        src_path = os.path.join('./data/tmp')
        dst_path = os.path.join(classes_root + '/', key)
        if os.path.exists(dst_path) == False:
            os.mkdir(dst_path)
        for file_name in samples[key]:
            src_file = os.path.join(src_path, file_name)
            dst_file = os.path.join(dst_path, file_name)
            shutil.copy(src_file, dst_file)
            print('Copying ' + src_file + ' to ' + dst_file)

    shutil.rmtree('./data/tmp')
    shutil.rmtree('./data/train')

    for folder_name in classes[:ratio]:
        source_path = os.path.join('./data/full-samples', folder_name)
        target_path = os.path.join('./data/cloud-samples', folder_name)
        print('Copying ' + source_path + ' to ' + target_path)
        shutil.copytree(source_path, target_path)
    for folder_name in classes[ratio:]:
        source_path = os.path.join('./data/full-samples', folder_name)
        target_path = os.path.join('./data/device-samples', folder_name)
        print('Copying ' + source_path + ' to ' + target_path)
        shutil.copytree(source_path, target_path)

    samples_per_class = math.floor((10 - ratio) * 5000 * 0.1 / ratio)
    for folder_name in classes[:ratio]:
        source_path = os.path.join('./data/cloud-samples/', folder_name)
        target_path = os.path.join('./data/device-enhanced-samples/',
                                   folder_name)
        os.makedirs(target_path, exist_ok=True)
        file_names = os.listdir(source_path)
        print('Selecting ' + str(samples_per_class) + ' samples from ' +
              source_path)
        selected_files = random.sample(file_names, samples_per_class)
        for file_name in selected_files:
            src_file = os.path.join(source_path, file_name)
            dst_file = os.path.join(target_path, file_name)
            print('Copying ' + src_file + ' to ' + dst_file)
            shutil.copy(src_file, dst_file)
    for folder_name in classes[ratio:]:
        source_path = os.path.join('./data/device-samples', folder_name)
        target_path = os.path.join('./data/device-enhanced-samples',
                                   folder_name)
        print('Copying ' + source_path + ' to ' + target_path)
        shutil.copytree(source_path, target_path)
    delete_per_folder = samples_per_class * ratio // (10 - ratio)
    if samples_per_class * ratio % (10 - ratio) != 0:
        delete_per_folder += 1
    files_to_delete = []
    for folder_name in classes[ratio:]:
        source_path = os.path.join('./data/device-enhanced-samples',
                                   folder_name)
        file_names = os.listdir(source_path)
        selected_files = random.sample(file_names, delete_per_folder)
        for file_name in selected_files:
            file_path = os.path.join(source_path, file_name)
            files_to_delete.append(file_path)
    random.shuffle(files_to_delete)
    for i, file_path in enumerate(files_to_delete):
        os.remove(file_path)
        if i >= (samples_per_class * ratio) - 1:
            break

    print('Done!')


def mean_std(root):
    print('Compute mean and variance for data.')
    # return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    dataset = CIFAR10(root=root, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return tuple(mean.numpy()), tuple(std.numpy())


def VGGG_make_layers(cfg, batch_norm=True):
    layers = []

    input_channel = 64
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    layers += [nn.Flatten()]

    return nn.Sequential(*layers)


def get_variable_name(var):
    for fi in reversed(inspect.stack()):
        names = [
            var_name for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            return names[0]


if __name__ == '__main__':
    download()
    extract()
    spilit()