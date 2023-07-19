import torch
from models import *
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


# 多卡训练后pth文件中的key会多一个module，需要处理一下
def process_pth(file_path):
    weights = torch.load(file_path)

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    return weights_dict


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=test_transform)
    testloader = DataLoader(testset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    criterion = torch.nn.CrossEntropyLoss()

    def test(shared_encoder, high_level_encoder, model, model_):
        shared_encoder.eval()
        high_level_encoder.eval()
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(testloader):
                x, label = x.to(device), label.to(device)

                x = shared_encoder(x)
                x_ = high_level_encoder(x)
                y = model(x_)
                y_ = model_(x)
                y = y.clone() + y_

                loss = criterion(y, label)

                test_loss += loss.item()
                _, predicted = y.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            acc = 100. * correct / total
            return acc

    vgg_shared_encoder = VGG_shared_encoder().to(device)
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/vgg_shared_encoder.pth'))
    vgg_control_model_mse = VGG_control_model().to(device)
    vgg_control_model_mse.load_state_dict(
        torch.load(r'./checkpoint/vgg_control_model_mse.pth'))

    dc_ccl_high_level_encoder = VGG_co_high_level_encoder().to(device)
    dc_ccl_high_level_encoder.load_state_dict(
        process_pth(r'./checkpoint/dc_ccl_high_level_encoder.pth'))
    dc_ccl_classifier = VGG_co_classifier().to(device)
    dc_ccl_classifier.load_state_dict(
        process_pth(r'./checkpoint/dc_ccl_classifier.pth'))
    dc_ccl_classifier_fine_tune = VGG_co_classifier().to(device)
    dc_ccl_classifier_fine_tune.load_state_dict(
        process_pth(r'./checkpoint/dc_ccl_classifier_fine_tune.pth'))

    dc_ccl_acc = test(vgg_shared_encoder,
                      dc_ccl_high_level_encoder,
                      model=dc_ccl_classifier,
                      model_=vgg_control_model_mse)
    print('DC-CCL accuracy: %.3f%%' % dc_ccl_acc)
    dc_ccl_fine_tune_acc = test(vgg_shared_encoder,
                                dc_ccl_high_level_encoder,
                                model=dc_ccl_classifier_fine_tune,
                                model_=vgg_control_model_mse)
    print('DC-CCL fine-tune accuracy: %.3f%%' % dc_ccl_fine_tune_acc)


if __name__ == '__main__':
    main()