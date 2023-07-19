import torch
import torch.optim as optim
import pickle
from models import *
from pipeline import Trainer


def get_result(file_root):
    with open(file_root, 'rb') as f:
        result = pickle.load(f)
    return result


# 多卡训练后pth文件中的key会多一个module，需要处理一下
def process_pth(file_path):
    weights = torch.load(file_path)

    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    return weights_dict


if __name__ == '__main__':
    cfg = {
        'dc_ccl_classifier_fine_tune': {
            'trainset_root': r'./data/full-samples',
            'testset_root': r'./data/test',
            'batch_size': 256,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 5e-3,
            'step_size': 2,
            'gamma': 0.5,
            'epochs': 10
        }
    }

    vgg_shared_encoder = VGG_shared_encoder()
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_shared_encoder) + '.pth'))
    vgg_control_model_mse = VGG_control_model()
    vgg_control_model_mse.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_control_model_mse) + '.pth'))

    dc_ccl_high_level_encoder = VGG_co_high_level_encoder()
    dc_ccl_high_level_encoder.load_state_dict(
        process_pth(r'./checkpoint/dc_ccl_high_level_encoder.pth'))
    dc_ccl_classifier_fine_tune = VGG_co_classifier()
    dc_ccl_classifier_fine_tune.load_state_dict(
        process_pth(r'./checkpoint/dc_ccl_classifier.pth'))

    dc_ccl_classifier_fine_tune_trainer = Trainer(
        cfg['dc_ccl_classifier_fine_tune'],
        shared_encoder=vgg_shared_encoder,
        high_level_encoder=dc_ccl_high_level_encoder,
        model=dc_ccl_classifier_fine_tune,
        model_=vgg_control_model_mse,
        mode='co-train',
        freeze=True,
        high_level_freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
    )
    dc_ccl_classifier_fine_tune_trainer.train()

    result_root = r'./result/'
    dc_ccl_classifier_fine_tune_result = get_result(
        result_root + utils.get_variable_name(dc_ccl_classifier_fine_tune) +
        '.pkl')
    print('after fine-tune accuracy: ',
          dc_ccl_classifier_fine_tune_result['accs'])
