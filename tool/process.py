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
from torch.autograd import Variable

from utils import utils_lr


def get_acc(preds, labels, converter):
    _, preds = preds.max(2)
    preds = preds.T
    all = 0
    t = 0
    for pred, label in zip(preds, labels):
        sim_pred = converter.decode(pred.data, raw=False)
        if sim_pred == label:
            t += 1
        all += 1
    return round(t / all, 2) * 100


def fit_epoch(gen, model, criterion, optimizer, converter, device, pbar):
    model = model.train()
    total_loss = 0
    total_accuracy = 0
    for iteration, batch in enumerate(gen):
        image, label = batch
        text, length = converter.encode(label)
        image = image.to(device)
        optimizer.zero_grad()
        preds = model(image)
        preds_size = torch.IntTensor([preds.shape[0]] * preds.shape[1])
        preds = preds.cpu()
        cost = criterion(preds, text, preds_size, length)
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


def val(gen, model, criterion, converter, device, pbar=None):
    model = model.eval()
    total_loss = 0
    total_accuracy = 0
    val_acc = 0
    for iteration, batch in enumerate(gen):
        image, label = batch
        text, length = converter.encode(label)
        image = image.to(device)
        preds = model(image)
        preds_size = torch.IntTensor([preds.shape[0]] * preds.shape[1])
        preds = preds.cpu()
        cost = criterion(preds, text, preds_size, length)

        acc = get_acc(preds, label, converter)
        total_accuracy += acc
        loss = cost.item()
        total_loss += loss
        val_acc = round(total_accuracy / (iteration + 1), 2)
        if pbar:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'val_acc': f"{val_acc}%",
            })
            pbar.update(1)
    return val_acc

