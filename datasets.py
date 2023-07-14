import torch
import os
from PIL import Image
import torchvision.transforms as transforms


class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.dataset = []
        self.transform = transform
        self.classes = [
            'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
            'ship', 'truck'
        ]

        for class_name in os.listdir(self.root):
            for file_name in os.listdir(os.path.join(self.root, class_name)):
                self.dataset.append(
                    (file_name, self.classes.index(class_name)))

    def __getitem__(self, index):
        file_name, label = self.dataset[index]
        file_path = self.root + '/' + self.classes[label] + '/' + file_name
        img = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010)),
    ])

    train = CIFAR10('./data/cloud-samples', transform=transform_train)
