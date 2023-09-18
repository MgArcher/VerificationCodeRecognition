# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : imizer.py
# Time       ：2023/9/15 10:03
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch
import torch.utils.data

from utils import utils_lr
from tool.imizer import get_acc

def fit_epoch(gen, model, criterion, optimizer,converter,device, teacher_model, criterion_kd, alpha, pbar=None):
    teacher_model = teacher_model.train()
    model = model.train()
    total_loss = 0
    total_accuracy = 0
    for iteration, batch in enumerate(gen):
        image, label = batch
        text, length = converter.encode(label)
        image = image.to(device)
        optimizer.zero_grad()
        teacher_preds = teacher_model(image)
        preds = model(image)
        preds_size = torch.IntTensor([preds.shape[0]] * preds.shape[1])
        preds = preds.cpu()
        teacher_preds = teacher_preds.cpu()
        cost = criterion(preds, text, preds_size, length)
        loss_soft = criterion_kd(preds.view(-1, 37), teacher_preds.view(-1, 37))
        cost = (1- alpha) * loss_soft + alpha * cost
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            acc = get_acc(preds, label, converter)
            total_accuracy += acc
            loss = cost.item()
            total_loss += loss
            if pbar:
                pbar.set_postfix(**{
                    'total_loss': total_loss / (iteration + 1),
                    'acc': f"{round(total_accuracy / (iteration + 1), 2)}%",
                    'lr': utils_lr.get_lr(optimizer)
                })
                pbar.update(1)


