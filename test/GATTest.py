import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from models.GAT import GAT
from utils.cell2attr import cell2attr, min_max_scaling
from utils.geom2edge_index import list2edge_index
from utils.write_output_csv import add_tensor_to_csv

if __name__ == '__main__':
    origin_data = cell2attr('../datasets/input/property')
    x = min_max_scaling(origin_data[:, 1:])
    y = origin_data[:, 0].long()
    print(y)
    print('---------输入特征-------------')
    print(x)
    edge_index = list2edge_index('../datasets/input/geom/cell_faces_num.dat',
                                 '../datasets/input/geom/cell_faces_indices.dat')
    # 生成train_mask和test_mask
    print('--------train_mask---------')
    # YX为5视为有矿，YX为6视为无矿
    print(y)
    y[(y != 5) & (y != 6)] = -1
    y[y == 5] = 1
    y[y == 6] = 0
    train_mask = (y == 1) | (y == 0)
    test_mask = y == -1

    print("Train mask values:", y[train_mask].shape)
    print("Test mask values:", y[test_mask].shape)
    # train_mask, val_mask, test_mask = gen_mask(x.shape[0])
    hidden_num = 4
    datasets = Data(x=x, edge_index=edge_index, y=y,
                    train_mask=train_mask, test_mask=test_mask)
    print('----------基本图结构------------')
    print(datasets)
    print(f'Number of nodes: {datasets.num_nodes}')
    print(f'Number of edges: {datasets.num_edges}')
    print(f'Average node degree: {datasets.num_edges / datasets.num_nodes:.2f}')
    print(f'Number of training nodes: {datasets.train_mask.sum()}')
    print(f'Training node label rate: {int(datasets.train_mask.sum()) / datasets.num_nodes:.2f}')
    print(f'Has isolated nodes: {datasets.has_isolated_nodes()}')
    print(f'Has self-loops: {datasets.has_self_loops()}')
    print(f'Is undirected: {datasets.is_undirected()}')

    model = GAT(datasets, hidden_num).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(datasets)
        # print(out[datasets.train_mask].shape)
        # print(datasets.y[datasets.train_mask].shape)
        # target = datasets.y[datasets.train_mask].double().unsqueeze(1)  # 解决维度不匹配问题
        loss = criterion(out[datasets.train_mask], 1 - datasets.y[datasets.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))

    with torch.no_grad():
        outputs = model(datasets)
        print('----------预测结果-----------')
        outputs = F.softmax(outputs, dim=1)
        predicted = outputs[:, 0]
        print(outputs)
        print(predicted)
        print(predicted.shape)
        add_tensor_to_csv(predicted, 'gat_result')
