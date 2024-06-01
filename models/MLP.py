import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size2, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x