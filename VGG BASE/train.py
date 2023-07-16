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
        'central_B': {
            'trainset_root': r'./data/full-samples',
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
        'cloud_B': {
            'trainset_root': r'./data/cloud-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-3,
            'step_size': 6,
            'gamma': 0.4,
            'epochs': 30,
        }
    }

    central_B = VGG_base_model()
    central_B_trainer = Trainer(
        cfg['central_B'],
        shared_encoder=None,
        model=central_B,
        model_=None,
        mode='co-train',
        freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    central_B_trainer.train()

    cloud_B = VGG_base_model()
    cloud_B_trainer = Trainer(
        cfg['cloud_B'],
        shared_encoder=None,
        model=cloud_B,
        model_=None,
        mode='co-train',
        freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    cloud_B_trainer.train()

    result_root = r'./result/'
    central_B_result = get_result(result_root +
                                  utils.get_variable_name(central_B) + '.pkl')
    cloud_B_result = get_result(result_root +
                                utils.get_variable_name(cloud_B) + '.pkl')
    print('central_B accuracy: ', central_B_result['accs'])
    print('cloud_B accuracy: ', cloud_B_result['accs'])