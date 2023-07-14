import pickle
import torch.optim as optim
from pipeline import Trainer
from models import *


def get_result(file_root):
    with open(file_root, 'rb') as f:
        result = pickle.load(f)
    return result


if __name__ == '__main__':
    cfg = {
        'cloud_model': {
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
            'epochs': 20,
        },
        'control_model': {
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
            'epochs': 20,
        },
        'co_submodel': {
            'trainset_root': r'./data/full-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 1,
            'gamma': 0.5,
            'epochs': 2,
        },
        'co_control_model': {
            'trainset_root': r'./data/full-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 1,
            'gamma': 0.5,
            'epochs': 2,
        },
        'decounpled_model': {
            'trainset_root': r'./data/full-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 1,
            'gamma': 0.5,
            'epochs': 2,
        }
    }
    shared_encoder = Shared_encoder()
    cloud_model = Cloud_model()
    cloud_model_trainer = Trainer(
        cfg['cloud_model'],
        shared_encoder=shared_encoder,
        model=cloud_model,
        mode='co-train',
        freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    cloud_model_trainer.train()

    control_model = Control_model()
    control_model_trainer = Trainer(
        cfg['control_model'],
        shared_encoder=shared_encoder,
        model=control_model,
        model_=cloud_model,
        mode='control-train',
        freeze=True,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
    )
    control_model_trainer.train()

    co_submodel = Co_submodel()
    co_submodel_trainer = Trainer(
        cfg['co_submodel'],
        shared_encoder=shared_encoder,
        model=co_submodel,
        model_=None,
        mode='co-train',
        freeze=True,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    co_submodel_trainer.train()

    co_control_model = Co_submodel()
    co_control_model_trainer = Trainer(
        cfg['co_control_model'],
        shared_encoder=shared_encoder,
        model=co_control_model,
        model_=control_model,
        mode='co-train',
        freeze=True,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    co_control_model_trainer.train()

    decounpled_model = Co_submodel()
    decounpled_model_trainer = Trainer(
        cfg['decounpled_model'],
        shared_encoder=shared_encoder,
        model=decounpled_model,
        model_=cloud_model,
        mode='co-train',
        freeze=True,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    decounpled_model_trainer.train()

    cloud_result = get_result(r'./result/cloud_model.pkl')
    control_result = get_result(r'./result/control_model.pkl')
    co_submodel_result = get_result(r'./result/co_submodel.pkl')
    co_control_model_result = get_result(r'./result/co_control_model.pkl')
    decounpled_model_result = get_result(r'./result/decounpled_model.pkl')
    print('cloud_model accuracy: ', cloud_result['accs'])
    print('control_model accuracy: ', control_result['accs'])
    print('co_submodel accuracy: ', co_submodel_result['accs'])
    print('co_control_model accuracy: ', co_control_model_result['accs'])
    print('decounpled_model accuracy: ', decounpled_model_result['accs'])