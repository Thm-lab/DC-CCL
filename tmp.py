import torch

cloud_model_pth = torch.load(r'./checkpoint/vgg_cloud_model.pth')

print(type(cloud_model_pth))

for cloud_model_pth_key in cloud_model_pth.keys():
    print(cloud_model_pth_key)