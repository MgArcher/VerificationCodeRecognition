# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : var.py
# Time       ：2023/9/15 11:17
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch.utils.data
from utils import utils
from utils.dataloader import CaptchaDataset, get_charactes_keys

from tool import load, imizer

class Opt():
    valRoot = r"data/jiandan_test"
    cuda = True
    pretrained = 'expr/best.pth'
    alphabet_path = 'utils/charactes_keys.txt'
    batchSize = 64
    nh = 256
    nc = 3
    workers = 0
    imgH = 32
    imgW = 100
    model_name = "crnnlite"


opt = Opt()
sampler = None
device=torch.device('cuda')
alphabet = get_charactes_keys(opt.alphabet_path)
test_dataset = CaptchaDataset(opt.valRoot, [opt.imgH, opt.imgW], alphabet, opt.nc)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    )

# 加载模型
model = load.load_model(opt, alphabet, opt.model_name)
# loss
criterion = torch.nn.CTCLoss()
# 解码器
converter = utils.strLabelConverter(alphabet)
val_acc = imizer.val(test_loader, model, criterion, converter, device)
print("val_acc:", val_acc)