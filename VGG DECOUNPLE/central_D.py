import torch
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
        'central_D': {
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
            'epochs': 2
        }
    }

    vgg_shared_encoder = VGG_shared_encoder()
    vgg_cloud_model = VGG_cloud_model()
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/' +
                     utils.get_variable_name(vgg_shared_encoder) + '.pth'))
    vgg_cloud_model.load_state_dict(
        torch.load(r'./checkpoint/' +
                        utils.get_variable_name(vgg_cloud_model) + '.pth'))

    central_D = VGG_co_submodel()
    central_D_trainer = Trainer(
        cfg['central_D'],
        shared_encoder=vgg_shared_encoder,
        model=central_D,
        model_=vgg_cloud_model,
        mode='co-train',
        freeze=True,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    central_D_trainer.train()

    result_root = r'./result/'
    central_D_result = get_result(result_root +
                                  utils.get_variable_name(central_D) + '.pkl')
    print('central_D accuracy: ', central_D_result['accs'])
