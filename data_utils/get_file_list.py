import os
import json
import random

def generate_file_list(root_dir):
    file_paths = []
    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 去掉.txt后缀
            filename = filename.replace('.txt', '')
            # 创建完整的文件路径
            full_path = os.path.join(dirpath, filename)
            # 添加到列表中，可以根据需要修改路径格式
            file_paths.append(full_path.replace("\\", "/"))  # 确保路径使用正斜杠
    return file_paths

def split_data(file_paths, train_ratio, val_ratio):
    # 随机打乱路径
    random.shuffle(file_paths)
    total = len(file_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    # 分割数据集
    train_files = file_paths[:train_end]
    val_files = file_paths[train_end:val_end]
    test_files = file_paths[val_end:]
    return train_files, val_files, test_files

def save_to_json(file_paths, json_file):
    # 将路径列表写入JSON文件
    with open(json_file, 'w') as f:
        json.dump(file_paths, f, indent=4)

# 定义根目录和输出文件名
root_directory = "data/teeth_segmentation_normal/00000001"
train_json = "data/teeth_segmentation_normal/train_test_split/train_files.json"
val_json = "data/teeth_segmentation_normal/train_test_split/val_files.json"
test_json = "data/teeth_segmentation_normal/train_test_split/test_files.json"

# 生成文件列表
paths = generate_file_list(root_directory)

# 分割数据集
train_paths, val_paths, test_paths = split_data(paths, 0.71, 0.145)  # 0.71, 0.145和0.145近似等于5:1:1的比例

# 保存到JSON
save_to_json(train_paths, train_json)
save_to_json(val_paths, val_json)
save_to_json(test_paths, test_json)

print(f"Train set saved with {len(train_paths)} entries.")
print(f"Validation set saved with {len(val_paths)} entries.")
print(f"Test set saved with {len(test_paths)} entries.")
