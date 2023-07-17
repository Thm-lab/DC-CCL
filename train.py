import pickle
import torch
import torch.optim as optim
from pipeline import Trainer
from models import *


def get_result(file_root):
    with open(file_root, 'rb') as f:
        result = pickle.load(f)
    return result


if __name__ == '__main__':
    cfg = {
        'vgg_cloud_model': {
            'trainset_root': r'./data/cloud-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-3,
            'step_size': 5,
            'gamma': 0.4,
            'epochs': 30,
        },
        'vgg_control_model_mse': {
            'trainset_root': r'./data/cloud-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 12,
            'gamma': 0.4,
            'epochs': 30,
        },
        'vgg_control_model_general': {
            'trainset_root': r'./data/cloud-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 14,
            'gamma': 0.4,
            'epochs': 30,
            'T': 20,
            'alpha': 0.1,
        },
    }

    vgg_shared_encoder = VGG_shared_encoder()
    vgg_cloud_model = VGG_cloud_model()
    vgg_cloud_model_trainer = Trainer(
        cfg['vgg_cloud_model'],
        shared_encoder=vgg_shared_encoder,
        model=vgg_cloud_model,
        model_=None,
        mode='co-train',
        freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    # vgg_cloud_model_trainer.train()
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/vgg_shared_encoder.pth'))
    vgg_cloud_model.load_state_dict(
        torch.load(r'./checkpoint/vgg_cloud_model.pth'))

    vgg_control_model_mse = VGG_control_model()
    vgg_control_model_mse_trainer = Trainer(
        cfg['vgg_control_model_mse'],
        shared_encoder=vgg_shared_encoder,
        model=vgg_control_model_mse,
        model_=vgg_cloud_model,
        mode='control-train',
        freeze=True,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
    )
    # vgg_control_model_mse_trainer.train()

    vgg_control_model_general = VGG_control_model()
    vgg_control_model_general_trainer = Trainer(
        cfg['vgg_control_model_general'],
        shared_encoder=vgg_shared_encoder,
        model=vgg_control_model_general,
        model_=vgg_cloud_model,
        mode='control-train',
        freeze=True,
        criterion=nn.CrossEntropyLoss,
        T=cfg['vgg_control_model_general']['T'],
        criterion_=nn.KLDivLoss,
        optimizer=optim.Adam,
    )
    vgg_control_model_general_trainer.train()

    result_root = r'./result/'
    vgg_cloud_model_result = get_result(
        result_root + utils.get_variable_name(vgg_cloud_model) + '.pkl')
    vgg_control_model_mse_result = get_result(
        result_root + utils.get_variable_name(vgg_control_model_mse) + '.pkl')
    vgg_control_model_general_result = get_result(
        result_root + utils.get_variable_name(vgg_control_model_general) +
        '.pkl')
    print('vgg_cloud_model accuracy: ', vgg_cloud_model_result['accs'])
    print('vgg_control_model_mse accuracy: ',
          vgg_control_model_mse_result['accs'])
    print('vgg_control_model_general accuracy: ',
          vgg_control_model_general_result['accs'])
