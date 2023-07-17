import os
import random

classes = [
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
]

k = 2
num = 1000

# 计算每个文件夹需要删除的文件数
num_per_folder = num // k
if num % k != 0:
    num_per_folder += 1

# 随机选择要删除的文件
files_to_delete = []
for folder_name in classes[-k:]:
    source_path = os.path.join('./data/device-enhanced-samples', folder_name)
    file_names = os.listdir(source_path)
    selected_files = random.sample(file_names, num_per_folder)
    for file_name in selected_files:
        file_path = os.path.join(source_path, file_name)
        files_to_delete.append(file_path)

# 删除文件
random.shuffle(files_to_delete)
for i, file_path in enumerate(files_to_delete):
    os.remove(file_path)
    if i >= num - 1:
        break