import torch
from models import *
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform_test)
    testloader = DataLoader(testset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    # shared_encoder = Shared_encoder()
    # shared_encoder = shared_encoder.to(device)
    # shared_encoder.load_state_dict(
    #     torch.load('./checkpoint/shared_encoder.pth'))
    # cloud_model = Cloud_model()
    # cloud_model = cloud_model.to(device)
    # cloud_model.load_state_dict(torch.load('./checkpoint/cloud_model.pth'))
    # control_model = Control_model()
    # control_model = control_model.to(device)
    # control_model.load_state_dict(torch.load('./checkpoint/control_model.pth'))
    # co_submodel = Co_submodel()
    # co_submodel = co_submodel.to(device)
    # co_submodel.load_state_dict(torch.load('./checkpoint/co_submodel.pth'))
    # co_control_model = Co_submodel()
    # co_control_model = co_control_model.to(device)
    # co_control_model.load_state_dict(
    #     torch.load('./checkpoint/co_control_model.pth'))
    # decounpled_model = Co_submodel()
    # decounpled_model = decounpled_model.to(device)
    # decounpled_model.load_state_dict(
    #     torch.load('./checkpoint/decounpled_model.pth'))
    vgg_base_model = VGG_base_model()
    vgg_base_model = vgg_base_model.to(device)
    vgg_base_model.load_state_dict(
        torch.load('./checkpoint/vgg_base_model.pth'))
    # vgg_shared_encoder = VGG_shared_encoder()
    # vgg_shared_encoder = vgg_shared_encoder.to(device)
    # vgg_shared_encoder.load_state_dict(
    #     torch.load('./checkpoint/vgg_shared_encoder.pth'))
    # vgg_cloud_model = VGG_cloud_model()
    # vgg_cloud_model = vgg_cloud_model.to(device)
    # vgg_cloud_model.load_state_dict(
    #     torch.load('./checkpoint/vgg_cloud_model.pth'))
    # vgg_control_model = VGG_control_model()
    # vgg_control_model = vgg_control_model.to(device)
    # vgg_control_model.load_state_dict(
    #     torch.load('./checkpoint/vgg_control_model.pth'))

    criterion = torch.nn.CrossEntropyLoss()

    def test(model, model_=None, mode=''):
        # shared_encoder.eval()
        # vgg_shared_encoder.eval()
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)

                if mode == '':
                    # x = shared_encoder(x)
                    pass
                elif mode == 'vgg':
                    # x = vgg_shared_encoder(x)
                    pass
                elif mode == 'base':
                    pass
                y = model(x)

                if model_ is not None:
                    model_.eval()
                    y_ = model_(x)
                    y += y_

                loss = criterion(y, label)

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            acc = 100. * correct / total
            return acc

    # print('==> Testing model..')
    # cloud_model_acc = test(cloud_model)
    # print('Cloud model accuracy: %.3f%%' % cloud_model_acc)
    # control_model_acc = test(control_model)
    # print('Control model accuracy: %.3f%%' % control_model_acc)
    # co_submodel_acc = test(co_submodel, cloud_model)
    # print('Co_submodel accuracy: %.3f%%' % co_submodel_acc)
    # co_control_model_acc = test(co_control_model, control_model)
    # print('Co_submodel + Control_model accuracy: %.3f%%' %
    #       co_control_model_acc)
    # decoupled_model_acc = test(decounpled_model, cloud_model)
    # print('Decoupled model accuracy: %.3f%%' % decoupled_model_acc)
    vgg_base_model_acc = test(vgg_base_model, mode='base')
    print('VGG base model accuracy: %.3f%%' % vgg_base_model_acc)
    # vgg_cloud_model_acc = test(vgg_cloud_model, mode='vgg')
    # print('VGG cloud model accuracy: %.3f%%' % vgg_cloud_model_acc)
    # vgg_control_model_acc = test(vgg_control_model, mode='vgg')
    # print('VGG control model accuracy: %.3f%%' % vgg_control_model_acc)


if __name__ == '__main__':
    main()