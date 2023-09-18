

# 使用流程
简单易用的验证码识别
包含整个流程

#数据集下载
数据集一 https://wwm.lanzoum.com/itczd0b5z3yj  
将压缩包放入到data中解压，在data目录下新建一个文件夹，剪切1000张图片到新文件夹中制作成验证集



#训练
修改train.py 中trainRoot和valRoot修改为你的训练集和验证集
```json
trainRoot = r"data/jiandan"  
valRoot = r"data/jiandan_test"  


# 自定义训练集
方式一

方式二


```
运行train.py

#推理
运行 var_torch.py
#部署
运行 export.py转换成onnx
运行 var_onnx.py

# 实现原理


# 参考文档
https://github.com/DayBreak-u/chineseocr_lite/tree/master
https://github.com/meijieru/crnn.pytorch