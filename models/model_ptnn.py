# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Time       ：2023/9/11 16:24
# Author     ：yujia
# version    ：python 3.6
# Description：pplcnet + Transformer
"""
import torch
import torch.nn as nn

from models.model_util import Transformer
from models.pplcnet import PPLCNet


class Model(nn.Module):
    def __init__(self,
                 num_class=37,
                 hidden_size=128,
                 ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.fe = PPLCNet()
        self.prediction = Transformer(self.num_class, hidden_size)

    def forward(self, x):
        x = self.fe(x)
        x = x.view(x.shape[0], -1, self.hidden_size)
        x = self.prediction(x)
        x = x.permute(1, 0, 2)
        return x


if __name__ == '__main__':
    model = Model()

    sample = torch.rand([4, 3, 32, 100])
    # print(sample)
    out = model(sample)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print(out.shape)
    torch.save(model.state_dict(), "1.pth")
