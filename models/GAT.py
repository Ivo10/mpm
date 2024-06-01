import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, data, hid_num):
        super(GAT, self).__init__()
        torch.manual_seed(123457)
        self.conv1 = GATConv(data.num_features, hid_num)
        self.conv2 = GATConv(hid_num, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
