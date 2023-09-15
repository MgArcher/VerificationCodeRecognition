# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ver_onnx.py.py
# Time       ：2023/3/29 14:20
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
from PIL import Image
import onnxruntime
import numpy as np
np.set_printoptions(precision=4)


class CaptchaONNX(object):
    def __init__(self, path, keys_path, providers):
        self.sess = onnxruntime.InferenceSession(path, providers=providers)
        self.input_shape = [32, 100]
        self.batch_size = 1
        self.character = self.get_charactes_keys(keys_path)

    def strLabelConverter(self, res, alphabet):
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(alphabet[res[i] - 1])
        return ''.join(raw)

    def open_image(self, file, input_shape):
        out = Image.open(file)
        # 改变大小 并保证其不失真
        out = out.convert('RGB')
        h, w = input_shape
        img_w, img_h = out.size
        widht = int(img_w * (h / img_h))
        out = out.resize((widht, h), Image.ANTIALIAS)
        return out

    def get_charactes_keys(self, path):
        charactes_keys = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                charactes_keys.append(line)
        return charactes_keys

    def reason(self, lines):
        image = self.open_image(lines, self.input_shape)
        image = np.array(image).astype(np.float32) / 255.0
        photo = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        out = self.sess.run(None, {"x1": photo})[0]
        pic = np.argmax(out, axis=2)
        pic = pic.T
        preds_str = self.strLabelConverter(pic[0], alphabet=self.character)
        return preds_str


if __name__ == '__main__':
    pre_onnx_path = "expr/best.onnx"
    keys_path = "utils/charactes_keys.txt"
    pre = CaptchaONNX(pre_onnx_path, keys_path=keys_path, providers=['CPUExecutionProvider'])
    img_path = "docs/35L3_1578456366900.jpg"
    large_img = pre.reason(img_path)
    print(large_img)
