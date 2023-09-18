# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : KD_train.py
# Time       ：2023/9/14 17:15
# Author     ：yujia
# version    ：python 3.6
# Description：知识蒸馏
"""
import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import os
from tqdm import tqdm

from utils import utils
from utils import dataloader
from utils import utils_lr
from tool import load, process, process_kd

class Opt():
    trainRoot = r"data/jiandan"
    valRoot = r"data/jiandan_test"
    cuda = True

    pretrained = ''
    alphabet_path = 'utils/charactes_keys.txt'
    expr_dir = 'expr'

    nepoch = 40
    nepoch_teacher = 10
    batchSize = 64
    nh = 256
    nc = 3
    workers = 0
    imgH = 32
    imgW = 100

    lr = 0.001
    beta1 = 0.5
    optimizer_type = "Adam"
    model_name = "crnn"
    manualSeed = 1234
    alpha = 0.1

opt = Opt()

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)
device=torch.device('cuda')
# 随机种子
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

# 训练集
alphabet = dataloader.get_charactes_keys(opt.alphabet_path)
train_dataset = dataloader.CaptchaDataset(opt.trainRoot, [opt.imgH, opt.imgW], alphabet, opt.nc)
sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    )
test_dataset = dataloader.CaptchaDataset(opt.valRoot, [opt.imgH, opt.imgW], alphabet, opt.nc)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    )

# 加载模型
teacher_model = load.load_model(opt, alphabet, "crnn")
model = load.load_model(opt, alphabet, "crnnlite")
# 加载学习器
optimizer = load.load_optimizer(opt, model)
lr_scheduler_func = utils_lr.get_lr_scheduler_func(opt.lr, opt.optimizer_type, opt.batchSize, opt.nepoch)

# loss
criterion = torch.nn.CTCLoss()
criterion_kd = torch.nn.KLDivLoss(reduction='batchmean')#KL散度
# 解码器
converter = utils.strLabelConverter(alphabet)



acc = 0
for epoch in range(1, opt.nepoch + 1):
    # 每代修改学习率
    utils_lr.set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    # 训练
    num_iterations = len(train_loader)
    pbar = tqdm(total=num_iterations, desc=f'Train student Epoch {epoch}/{opt.nepoch}', postfix=dict, mininterval=0.3)
    process_kd.fit_epoch(train_loader, model, criterion, optimizer, converter, device, teacher_model, criterion_kd, opt.alpha, pbar)
    pbar.close()
    # 验证
    num_val = len(test_loader)
    pbar = tqdm(total=num_val, desc=f'Validation student Epoch {epoch}/{opt.nepoch}', postfix=dict, mininterval=0.3)
    val_acc = process.val(test_loader, model, criterion, converter, device, pbar)
    pbar.close()
    # 保存模型
    torch.save(model.state_dict(), os.path.join(opt.expr_dir, f"model_{epoch}.pth"))
    if val_acc >= acc:
        torch.save(model.state_dict(), os.path.join(opt.expr_dir, f"best.pth"))
        acc = val_acc