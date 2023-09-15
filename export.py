# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : export.py.py
# Time       ：2023/3/29 14:23
# Author     ：yujia
# version    ：python 3.6
# Description：
"""

# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ver.py
# Time       ：2023/3/22 16:17
# Author     ：yujia
# version    ：python 3.6
# Description：
"""
import torch
import models.crnn as crnn
from models.crnn_lite import CrnnLite


class Opt():
    model_path = "expr/best.pth"
    export_onnx_file = "expr/best.onnx"
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    nh = 256
    nc = 3
    workers = 0
    imgH = 32
    imgW = 100
    lite_model = True


opt = Opt()
sample = torch.rand([1, opt.nc, opt.imgH, opt.imgW])
model_path = opt.model_path
export_onnx_file = opt.export_onnx_file
alphabet = opt.alphabet
device = torch.device('cpu')
nclass = len(alphabet) + 1
if opt.lite_model:
    model = CrnnLite(opt.imgH, opt.nc, nclass, opt.nh)
else:
    model = crnn.CRNN(opt.imgH, opt.nc, nclass, opt.nh)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.eval()


torch.onnx.export(
    model,
    sample,
      export_onnx_file,
      opset_version=13,
      do_constant_folding=True,  # 是否执行常量折叠优化
      input_names=["x1"],  # 输入名
      output_names=["output"],  # 输出名
      dynamic_axes={
          "x1": {
              0: "batch_size",
              3: "imgw",
          },  # 批处理变量
      }
      )
