import torch
import torch.nn as nn


class SimCLR_Model(nn.Module):

    def __init__(self, base_encoder):
        super(SimCLR_Model, self).__init__()
        self.base_encoder = base_encoder
        self.base_encoder.fc = nn.Identity()
        self.projection_head = nn.Sequential(
          nn.Linear(64, 64, bias=False),
          nn.ReLU(),
          nn.Linear(64, 128, bias=False)
        )

    def forward(self, input_batch):
        h1 = self.base_encoder(input_batch)
        z1 = self.projection_head(h1)
        return h1, z1
