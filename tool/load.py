# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : load.py
# Time       ：2023/9/15 9:47
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm


from utils import utils
from utils.dataloader import CaptchaDataset, get_charactes_keys
import models.crnn as crnn
from models.crnn_lite import CrnnLite
from utils import utils_lr


def load_model(opt, alphabet, model_name):
    device = torch.device('cuda')
    # 模型
    nclass = len(alphabet) + 1

    if model_name == "crnnlite":
        model = CrnnLite(opt.imgH, opt.nc, nclass, opt.nh)
    elif model_name == "crnn":
        model = crnn.CRNN(opt.imgH, opt.nc, nclass, opt.nh)
    else:
        model = crnn.CRNN(opt.imgH, opt.nc, nclass, opt.nh)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        state_dict = torch.load(opt.pretrained)
        model.load_state_dict(state_dict, strict=False)
    print(model)
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