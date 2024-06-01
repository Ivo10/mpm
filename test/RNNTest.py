import numpy as np
import torch
import torch.nn as nn

from models.LSTM import LSTM
from utils.cell2attr import cell2attr, min_max_scaling, split_train_valid_test
from utils.gen_res_dat import gen_res_dat
from utils.write_output_csv import add_tensor_to_csv, write_content
from utils.plot import plot_loss_vs_epoch
from utils.val import calculate_accuracy

origin_data = cell2attr('../datasets/input/property')
print('----------原始数据-----------')
print(origin_data)


# 划分训练集、验证集、测试集
train_x, train_y, val_x, val_y, test_x = split_train_valid_test(origin_data)
# 归一化train_x、test_x
train_x = min_max_scaling(train_x)
test_x = min_max_scaling(test_x)
val_x = min_max_scaling(val_x)

# YX为5定义为有矿，YX为6定义为无矿
train_y[train_y == 5] = 1
train_y[train_y == 6] = 0
train_y = train_y.unsqueeze(1)
train_y = torch.cat((train_y, 1 - train_y), dim=1)
val_y[val_y == 5] = 1
val_y[val_y == 6] = 0
val_y = val_y.unsqueeze(1)
val_y = torch.cat((val_y, 1 - val_y), dim=1)
print('----------训练集属性-----------')
print(train_x)
print(train_x.shape)
print(val_x)
print(val_x.shape)

input_size = 1  # 输入特征数
hidden_size = 10  # 隐藏层特征数
num_layers = 5  # LSTM层数
output_size = 2  # 输出类别数
batch_size = train_x.shape[0]  # 批大小（整体输入）
sequence_length = train_x.shape[1]  # LSTM序列长度（输入属性个数）

# 数据预处理，整理为(sequence_length, batch_size, input_size)
train_x = np.transpose(train_x)
train_x = train_x[:, :, np.newaxis].float()
val_x = np.transpose(val_x)
val_x = val_x[:, :, np.newaxis].float()
test_x = np.transpose(test_x)
test_x = test_x[:, :, np.newaxis].float()

# 定义优化器和损失函数
model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
epochs = []
losses = []
for epoch in range(num_epochs):
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        val_result = model(val_x)
        accuracy = calculate_accuracy(val_result, val_y)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, loss.item(), accuracy))

    epochs.append(epoch)
    losses.append(loss.item())

plot_loss_vs_epoch(losses, epochs)

# 预测新数据
with torch.no_grad():
    outputs = model(test_x)
    print('----------预测结果-----------')
    predicted = outputs[:, 0]
    print(outputs)
    print(predicted)
    print(predicted.shape)
    write_content('../datasets/input/property',
                  '../datasets/output')
    gen_res_dat(predicted[:, np.newaxis], '../datasets/output/rnn.dat')
    add_tensor_to_csv(predicted, 'rnn_result')
