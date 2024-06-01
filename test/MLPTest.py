import torch
import torch.nn as nn

from models.MLP import MLP
from utils.cell2attr import cell2attr, min_max_scaling
from utils.write_output_csv import add_tensor_to_csv

origin_data = cell2attr('../datasets/input/property')
print('----------原始数据-----------')
print(origin_data)

# 制作训练数据
train_set = origin_data[(origin_data[:, 0] == 5) | (origin_data[:, 0] == 6)]
train_x = train_set[:, 1:].float()
test_x = origin_data[:, 1:].float()

# 归一化train_x、test_x
train_x = min_max_scaling(train_x)
test_x = min_max_scaling(test_x)

train_y = train_set[:, 0]
train_y[train_y == 5] = 1
train_y[train_y == 6] = 0
train_y = train_y.unsqueeze(1)
train_y = torch.cat((train_y, 1 - train_y), dim=1)
print('----------归一化后属性-----------')
print(train_x)

hidden_size1 = 10  # 隐藏层特征数
hidden_size2 = 5
output_size = 2  # 输出类别数

# 定义优化器和损失函数
model = MLP(train_x.shape[1], hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 预测新数据
with torch.no_grad():
    outputs = model(test_x)
    print('----------预测结果-----------')
    predicted = outputs[:, 0]
    print(outputs)
    print(predicted)
    print(predicted.shape)
    add_tensor_to_csv(predicted, 'mlp_result')
