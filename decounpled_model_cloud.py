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
        'decounpled_model_cloud': {
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
        'decounpled_model_device': {
            'trainset_root': r'./data/data/device-enhanced-samples',
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
    rank = local_rank + node_rank * local_size
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl',
                                         init_method='tcp://{}:{}'.format(
                                             args.master_addr,
                                             args.master_port),
                                         rank=rank,
                                         world_size=world_size)

    vgg_shared_encoder = VGG_shared_encoder()
    vgg_cloud_model = VGG_cloud_model()
    vgg_shared_encoder.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_shared_encoder) + '.pth'))
    vgg_cloud_model.load_state_dict(
        torch.load(r'./checkpoint/' +
                   utils.get_variable_name(vgg_cloud_model) + '.pth'))

    decounpled_model_cloud = VGG_co_submodel()
    decounpled_model_cloud_trainer = Trainer(cfg['decounpled_model_cloud'],
                                            shared_encoder=vgg_shared_encoder,
                                            model=decounpled_model_cloud,
                                            model_=vgg_cloud_model,
                                            mode='co-train',
                                            freeze=True,
                                            criterion=nn.CrossEntropyLoss,
                                            optimizer=optim.SGD,
                                            ddp=True,
                                            local_rank=local_rank)
    decounpled_model_cloud_trainer.train()


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
