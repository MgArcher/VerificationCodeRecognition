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
from tool.load import load_model


class Opt():
    pretrained = "expr/best_expr.pth"
    export_onnx_file = "expr/best_expr.onnx"
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    nh = 256
    nc = 3
    workers = 0
    imgH = 32
    imgW = 100
    lite_model = True
    model_name = 'crnnlite'
    cuda = False


opt = Opt()
sample = torch.rand([1, opt.nc, opt.imgH, opt.imgW])
export_onnx_file = opt.export_onnx_file
alphabet = opt.alphabet
nclass = len(alphabet) + 1
model = load_model(opt, alphabet, opt.model_name)
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
