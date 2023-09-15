# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model.py
# Time       ：2023/9/11 16:24
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch
import torch.nn as nn

from models.model_util import TPS_STN, ResNet50, BiLSTM, Transformer


class Model(nn.Module):
    def __init__(self,
                 imgh=32,
                 imgw=100,
                 input_channel=3,
                 output_channel=512,
                 hidden_size=256,
                 num_fiducial=20,
                 num_class=37,
                 bilstm=True,
                 device=torch.device('cpu')):
        super(Model, self).__init__()

        self.num_class = num_class
        self.bilstm = bilstm

        self.transformation = TPS_STN(num_fiducial, I_size=(imgh, imgw), I_r_size=(imgh, imgw), device=device,
                                      I_channel_num=input_channel)
        self.fe = ResNet50(input_channel, output_channel)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.seq = nn.Sequential(BiLSTM(output_channel, hidden_size, hidden_size),
                                 BiLSTM(hidden_size, hidden_size, hidden_size))
        if self.bilstm:
            self.seq_out_channels = hidden_size
        else:
            print('没有指定序列模型')
            self.seq_out_channels = output_channel
        self.prediction = Transformer(self.num_class, self.seq_out_channels)

    def forward(self, x):
        x = self.transformation(x)
        x = self.fe(x)
        x = self.adaptive_avg_pool(x.permute(0,3,1,2))  # [b, c, h, w] -> [b, w, c, h]
        x = x.squeeze(3)
        if self.bilstm:
            x = self.seq(x)
        pred = self.prediction(x.contiguous())
        return pred

if __name__ == '__main__':
    model = Model()

    sample = torch.rand([1, 3, 32, 300])
    out = model(sample)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print(out.shape)
