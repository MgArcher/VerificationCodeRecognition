# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ver_torch.py
# Time       ：2023/9/13 11:18
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch
import numpy as np
from PIL import Image

from utils.dataloader import get_charactes_keys
from utils import utils
from tool import load

class Opt():
    cuda = False
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
alphabet = get_charactes_keys(opt.alphabet_path)
model = load.load_model(opt, alphabet, opt.model_name)
model = model.eval()

converter = utils.strLabelConverter(alphabet)


def open_image(file, input_shape):
    out = Image.open(file)
    # 改变大小 并保证其不失真
    out = out.convert('RGB')
    h, w = input_shape
    img_w, img_h = out.size
    widht = int(img_w * (h / img_h))
    out = out.resize((widht, h), Image.ANTIALIAS)
    return out


def reason(lines):
    input_shape = [32, 80]
    image = open_image(lines, input_shape)
    image = np.array(image).astype(np.float32) / 255.0
    photo = torch.from_numpy(np.expand_dims(np.transpose(image, (2, 0, 1)), 0)).type(torch.FloatTensor)
    preds = model(photo)
    _, preds = preds.max(2)
    preds = preds.T
    for pred in preds:
        raw_pred = converter.decode(pred.data, raw=True)
        sim_pred = converter.decode(pred.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))


if __name__ == '__main__':
    img_path = "docs/35L3_1578456366900.jpg"
    preds_str = reason(img_path)