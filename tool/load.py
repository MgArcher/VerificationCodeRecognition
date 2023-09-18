# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : load.py
# Time       ：2023/9/15 9:47
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

import torch

import torch.optim as optim
import torch.utils.data

import models.crnn as crnn
from models.crnn_lite import CRnn
from models.model_svtr import Model as SVTRModel
from models.model_ptnn import Model as PTNNMode


# 在CRNN上调用自定义权重初始化 ***很重要
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_model(opt, alphabet, model_name):
    if opt.cuda is True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # 模型
    nclass = len(alphabet) + 1

    if model_name == "crnnlite":
        model = CRnn(opt.imgH, opt.nc, nclass, opt.nh)
        model.apply(weights_init)
    elif model_name == "crnn":
        model = crnn.CRNN(opt.imgH, opt.nc, nclass, opt.nh)
        model.apply(weights_init)
    elif model_name == "svtr":
        model = SVTRModel(imgh=opt.imgH, num_class=nclass, input_channel=opt.nc, device=device)
    elif model_name == "ptnn":
        model = PTNNMode(num_class=nclass, hidden_size=opt.nh, device=device)
    else:
        model = crnn.CRNN(opt.imgH, opt.nc, nclass, opt.nh)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        state_dict = torch.load(opt.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    # print(model)
    model = model.to(device)
    return model


def load_optimizer(opt,model):
    if opt.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.optimizer_type == "Adadelta":
        optimizer = optim.Adadelta(model.parameters())
    elif opt.optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
    elif opt.optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    else:
        optimizer = optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    return optimizer