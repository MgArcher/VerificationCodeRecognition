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

import models.crnn as crnn
from utils import utils


model_path = r'C:\Users\Administrator\Desktop\fsdownload\model_36.pth'
img_path = "docs/IN33.jpg"
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
device = torch.device('cpu')
imgH = 32
nc = 3
nclass = len(alphabet) + 1
nh = 256
input_shape = [imgH, 80]

model = crnn.CRNN(32, 3, len(alphabet) + 1, 256)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
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
    preds_str = reason(img_path)
    print(preds_str)