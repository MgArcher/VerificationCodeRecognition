# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : crnn_lite.py.py
# Time       ：2023/9/14 17:04
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CrnnLite(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, lstmFlag=True):
        """
        是否加入lstm特征层
        """
        super(CrnnLite, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [5, 3, 3, 3, 3, 3, 2]
        ps = [2, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        ss = [2, 1, 1, 1, 1, 1, 1]
        nm = [24, 128, 256, 256, 512, 512, 512]
        # nm = [32, 64, 128, 128, 256, 256, 256]
        # exp_ratio = [2,2,2,2,1,1,2]
        self.lstmFlag = lstmFlag

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            # exp  = exp_ratio[i]
            # exp_num = exp * nIn
            if i == 0:
                cnn.add_module('conv_{0}'.format(i),
                               nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                cnn.add_module('relu_{0}'.format(i), nn.ReLU(True))
            else:

                cnn.add_module('conv{0}'.format(i),
                               nn.Conv2d(nIn, nIn, ks[i], ss[i], ps[i], groups=nIn))
                if batchNormalization:
                    cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nIn))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

                cnn.add_module('convproject{0}'.format(i),
                               nn.Conv2d(nIn, nOut, 1, 1, 0))
                if batchNormalization:
                    cnn.add_module('batchnormproject{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        if self.lstmFlag:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(nm[-1], nh // 2, nh),
                BidirectionalLSTM(nh, nh // 4, nclass)
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(nm[-1], nh // 2),
                nn.Linear(nh // 2, nclass),
            )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print(conv.shape)
        assert h == 1, "the height of conv must be 1"
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        if self.lstmFlag:
            # rnn features
            output = self.rnn(conv)
        else:
            T, b, h = conv.size()

            t_rec = conv.contiguous().view(T * b, h)

            output = self.linear(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

        return output

if __name__ == '__main__':
    import torch
    imgH=32
    nc=3
    nclass=36
    nh= 256
    crnn = CrnnLite(imgH, nc, nclass, nh)
    img = torch.rand([4, 3, 32, 100])
    x = crnn(img)
    print(x.shape)