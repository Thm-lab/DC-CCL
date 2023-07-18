import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from pipeline import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--world_size', type=int)
parser.add_argument('--node_rank', type=int)
parser.add_argument('--master_addr', default='127.0.0.1', type=str)
parser.add_argument('--master_port', default='12355', type=str)
args = parser.parse_args()


def main(local_rank, node_rank, local_size, world_size):
    cfg = {
        'dc_ccl_cloud': {
            'trainset_root': r'./data/cloud-samples',
            'testset_root': r'./data/test',
            'batch_size': 500,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 2,
            'gamma': 0.5,
            'epochs': 2,
        },
        'dc_ccl_device': {
            'trainset_root': r'./data/device-enhanced-samples',
            'testset_root': r'./data/test',
            'batch_size': 500,
            'shuffle': True,
            'num_workers': 2,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0,
            'step_size': 2,
            'gamma': 0.5,
            'epochs': 2,
        }
    }
    rank = local_rank + node_rank * local_size
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_addr,
                                             args.master_port),
                                         rank=rank,
                                         world_size=world_size)

    vgg_shared_encoder = VGG_shared_encoder()
    vgg_control_model_mse = VGG_co_submodel()
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_shared_encoder) + '.pth'))
    vgg_control_model_mse.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_control_model_mse) + '.pth'))

    dc_ccl_high_level_encoder_device = VGG_co_high_level_encoder()
    dc_ccl_clasifier_device = VGG_co_classifier()
    dc_ccl_cloud_trainer = Trainer(
        cfg['dc_ccl_device'],
        shared_encoder=vgg_shared_encoder,
        high_level_encoder=dc_ccl_high_level_encoder_device,
        model=dc_ccl_clasifier_device,
        model_=vgg_control_model_mse,
        mode='co-train',
        freeze=True,
        high_level_freeze=False,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        ddp=True,
        local_rank=local_rank)
    dc_ccl_cloud_trainer.train()


if __name__ == "__main__":
    local_size = torch.cuda.device_count()
    print("local_size: %s" % local_size)
    torch.multiprocessing.spawn(main,
                                args=(
                                    args.node_rank,
                                    local_size,
                                    args.world_size,
                                ),
                                nprocs=local_size,
                                join=True)
