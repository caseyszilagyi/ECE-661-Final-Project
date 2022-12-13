import torch
import torch.nn as nn


class SimCLR_Model(nn.Module):
    def __init__(self, base_encoder):
        super(SimCLR_Model, self).__init__()
        self.base_encoder = base_encoder
        self.base_encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
          nn.Linear(64, 2048, bias=False),
          nn.ReLU(),
          nn.Linear(2048, 128, bias=False)
        )

    def forward(self, input_batch):
        h1 = self.base_encoder(input_batch)
        z1 = self.projection_head(h1)
        return h1, z1

class Linear_Eval(nn.Module):
    def __init__(self, base):
        super(Linear_Eval, self).__init__()
        self.base = base
        self.linear_layer = nn.Linear(64, 10, bias=True)

    def forward(self, input_batch):
        h = self.base(input_batch)
        z = self.linear_layer(h)
        return z

