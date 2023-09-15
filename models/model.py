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
from models.pplcnet import PPLCNet_x0_25


class Model(nn.Module):
    def __init__(self,
                 max_char=8,
                 num_class=39,
                 hidden_size=128):
        super(Model, self).__init__()

        self.num_class = num_class
        self.fe = PPLCNet_x0_25(num_classes=max_char * hidden_size)
        self.view = lambda x: x.view(x.shape[0], max_char, hidden_size)
        self.prediction = Transformer(self.num_class, hidden_size)

    def forward(self, x):
        x = self.fe(x)
        x = self.view(x)
        x = self.prediction(x.contiguous())
        # _, preds_index = x.max(2)
        return x


if __name__ == '__main__':
    model = Model()

    sample = torch.rand([8, 3, 32, 100])
    print(sample)
    out = model(sample)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    print(out.shape)
    print(preds_index.shape)
    print(preds_index)
    length_for_pred = torch.IntTensor([0] * 8)
    print(length_for_pred)

    device = torch.device('cpu')
    import string

    punctuation = r"""'.-"""
    character = string.digits + string.ascii_lowercase + punctuation
    from utils.misc import TransLabelConverter
    converter = TransLabelConverter(character, device)
    preds_str = converter.decode(preds_index, length_for_pred)
    print(preds_str)
    # labels = converter.decode(text_for_loss, length_for_loss)