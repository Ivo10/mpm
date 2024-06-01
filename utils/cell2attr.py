import os
import struct

import numpy as np
import torch


# 获取tensor格式特征矩阵
def cell2attr(folder_path):
    print('----------读取属性中---------------')
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    for file in file_paths:
        print(file)
    headers = {}
    datas = []
    for file_path in file_paths:
        headers[file_path] = parse_file(file_path)
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = []
            dtype, nbytes = headers[file_path]['type'], headers[file_path]['nbytes']
            file.seek(128)
            while True:
                item = file.read(int(nbytes))
                if not item:
                    break
                item = struct.unpack(data_map()[dtype], item)[0]

                data.append(item)
        datas.append(data)
    data_matrix = np.array(datas).T
    return torch.from_numpy(data_matrix).double()


# 数据转换格式对照
def data_map():
    return {'int': 'I', 'double': 'd', 'float': 'f', 'short': 'h'}


# 解析文件头，以字典形式返回
def parse_file(file):
    header_info = {}
    with open(file, 'rb') as file:
        for _ in range(7):
            line = file.readline().strip()
            if line:
                key, value = line.split(b"=")
                header_info[key.strip().decode('utf-8')] = value.strip().decode('utf-8')
    return header_info


# 数据归一化：最大最小缩放
def min_max_scaling(x):
    min_val = torch.min(x, dim=0, keepdim=True).values
    max_val = torch.max(x, dim=0, keepdim=True).values
    x = (x - min_val) / (max_val - min_val)
    return x


# 划分训练集、验证集、测试集合（输入所有特征x，第一列是标签）
def split_train_valid_test(x):
    train_set5 = x[x[:, 0] == 5]
    train_set6 = x[x[:, 0] == 6]
    train_proportion5 = (int)(train_set5.shape[0] * 0.9)
    train_proportion6 = (int)(train_set6.shape[0] * 0.9)
    train_x5 = train_set5[:train_proportion5, 1:]
    val_x5 = train_set5[train_proportion5:, 1:]
    train_y5 = train_set5[:train_proportion5, 0]
    val_y5 = train_set5[train_proportion5:, 0]
    train_x6 = train_set6[:train_proportion6, 1:]
    val_x6 = train_set6[train_proportion6:, 1:]
    train_y6 = train_set6[:train_proportion6, 0]
    val_y6 = train_set6[train_proportion6:, 0]

    train_x = torch.cat([train_x5, train_x6], 0)
    val_x = torch.cat([val_x5, val_x6], 0)
    train_y = torch.cat([train_y5, train_y6], 0)
    val_y = torch.cat([val_y5, val_y6], 0)
    test_x = x[:, 1:]  # 测试集取全部

    return train_x, train_y, val_x, val_y, test_x


if __name__ == '__main__':
    print(parse_file('../datasets/input/property/Property_13.dat'))
    attr = cell2attr('../datasets/input/property/')

    print(attr.shape)
    print(attr)
