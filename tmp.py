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
        }
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
    vgg_cloud_model_trainer.train()

    vgg_cloud_model_result = get_result(
        r'./result/' + utils.get_variable_name(vgg_cloud_model) + '.pkl')
    print('vgg_base_model accuracy: ', vgg_cloud_model_result['accs'])
