
# 超简单易用的验证码识别
**示例图片:**
![Example Image](./docs/AQQH_1578452834528.png)  
**识别结果：aqqh，推理耗时：11.97ms**
# 免责声明
**本项目旨在研究深度学习在验证码攻防上的应用。仅供学习交流使用，请勿用于非法用途，不得在任何商业使用，本人不承担任何法律责任。**
# 请作者喝可乐**o(*￣︶￣*)o**
![Example Image](./docs/img_3.png)  
# 使用流程
#### 环境准备
```shell
python3.6以上  
pip install -r requirements.txt
```
#### 推理
运行方式：
```shell
python var_torch.py
```
参数调整：
```python
class Opt():
    cuda = False
    pretrained = 'expr/best_expr.pth' # 选择的模型
    alphabet_path = 'tool/charactes_keys.txt' # 选择的字典

if __name__ == '__main__':
    img_path = "docs/AQQH_1578452834528.png" # 需要识别的图片
```
运行结果：
```
loading pretrained model from expr/best_expr.pth
----aa----qq----qq---hh------ => aqqh                
识别结果：aqqh，推理耗时：31.91ms
```
#### onnx转换
运行方式：
```shell
python export.py
```
参数调整：
```python
class Opt():
    pretrained = "expr/best_expr.pth" # 需要转换的模型
    export_onnx_file = "expr/best_expr.onnx" # 转换后的模型
    alphabet_path = 'tool/charactes_keys.txt'# 选择的字典
```
运行结果：
```
============== Diagnostic Run torch.onnx.export version 2.0.0+cpu ==============
verbose: False, log level: Level.ERROR
======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

save model in : expr/best_expr.onnx
```
将会在指定文件夹中看到生成了一个.onnx文件
#### onnx推理
运行方式：
```shell
python var_onnx.py
```
参数调整：
```python
if __name__ == '__main__':
    pre_onnx_path = "expr/best_expr.onnx"# 选择的模型
    keys_path = "tool/charactes_keys.txt"# 选择的字典
    pre = CaptchaONNX(pre_onnx_path, keys_path=keys_path, providers=['CPUExecutionProvider'])
    img_path = "docs/AQQH_1578452834528.png" # 需要识别的图片
    s = time.time()
    preds_str = pre.reason(img_path) # 识别函数
    print(f"识别结果：{preds_str}，推理耗时：{round((time.time() - s)*1000, 2)}ms")
```
运行结果：
```
识别结果：aqqh，推理耗时：8.0ms
```
#### 数据集
[数据集一](https://wwm.lanzoum.com/itczd0b5z3yj)

#### 训练
运行方式：
```shell
python train.py
```
在运行前修改 train.py 中 trainRoot 和 valRoot ，将其修改为你下载的训练集和验证集。  
可根据上方给的数据据下载，将压缩包放入到data中解压，在data目录下新建一个文件夹，
剪切1000张图片到新文件夹中制作成验证集，其余数据作为训练集
```python
class Opt():
    trainRoot = r"data/jiandan"  # 修改成你放入到data目录下的文件夹名称 这是训练集路径
    valRoot = r"data/jiandan_test" # 修改成你放入到data目录下的文件夹名称 这是测试集路径
    cuda = True #是否使用gpu
    pretrained = '' #模型继续训练 传入空则从头开始训练
    alphabet_path = 'tool/charactes_keys.txt'#选择的字典
    expr_dir = 'expr'#模型保存目录
```
运行结果：
```
Train Epoch 1/100: 100%|█████████████████| 119/119 [00:09<00:00, 12.26it/s, acc=0.0%, lr=0.0002, total_loss=1.24]
Validation Epoch 1/100: 100%|█████████████████████| 16/16 [00:00<00:00, 21.03it/s, total_loss=1.47, val_acc=0.0%]
Train Epoch 2/100: 100%|█████████████████| 119/119 [00:08<00:00, 14.28it/s, acc=0.0%, lr=0.0005, total_loss=2.33]
Validation Epoch 2/100: 100%|█████████████████████| 16/16 [00:00<00:00, 22.45it/s, total_loss=2.84, val_acc=0.0%]
Train Epoch 3/100: 100%|██████████████████| 119/119 [00:08<00:00, 14.38it/s, acc=0.0%, lr=0.001, total_loss=3.29]
Validation Epoch 3/100: 100%|█████████████████████| 16/16 [00:00<00:00, 22.12it/s, total_loss=3.57, val_acc=0.0%]
Train Epoch 4/100: 100%|██████████████████| 119/119 [00:08<00:00, 14.43it/s, acc=0.0%, lr=0.001, total_loss=3.67]
Validation Epoch 4/100: 100%|█████████████████████| 16/16 [00:00<00:00, 21.15it/s, total_loss=3.78, val_acc=0.0%]
Train Epoch 5/100: 100%|███████████████| 119/119 [00:08<00:00, 14.73it/s, acc=0.0%, lr=0.000999, total_loss=3.79]
Validation Epoch 5/100: 100%|█████████████████████| 16/16 [00:00<00:00, 21.95it/s, total_loss=3.85, val_acc=0.0%]
Train Epoch 6/100: 100%|███████████████| 119/119 [00:08<00:00, 14.36it/s, acc=0.0%, lr=0.000997, total_loss=3.82]
Validation Epoch 6/100: 100%|█████████████████████| 16/16 [00:00<00:00, 21.00it/s, total_loss=3.89, val_acc=0.0%]
Train Epoch 7/100: 100%|███████████████| 119/119 [00:08<00:00, 14.43it/s, acc=0.0%, lr=0.000995, total_loss=3.84]
Validation Epoch 7/100: 100%|█████████████████████| 16/16 [00:00<00:00, 21.14it/s, total_loss=3.89, val_acc=0.0%]
Train Epoch 8/100: 100%|███████████████| 119/119 [00:08<00:00, 14.54it/s, acc=0.0%, lr=0.000993, total_loss=3.84]
Validation Epoch 8/100: 100%|█████████████████████| 16/16 [00:00<00:00, 22.10it/s, total_loss=3.91, val_acc=0.0%]
Train Epoch 9/100: 100%|████████████████| 119/119 [00:08<00:00, 14.65it/s, acc=0.0%, lr=0.00099, total_loss=3.83]
Validation Epoch 9/100: 100%|█████████████████████| 16/16 [00:00<00:00, 22.07it/s, total_loss=3.88, val_acc=0.0%]
Train Epoch 10/100: 100%|██████████████| 119/119 [00:08<00:00, 14.73it/s, acc=0.0%, lr=0.000986, total_loss=3.82]
Validation Epoch 10/100: 100%|████████████████████| 16/16 [00:00<00:00, 21.11it/s, total_loss=3.85, val_acc=0.0%]
Train Epoch 11/100: 100%|███████████████| 119/119 [00:08<00:00, 14.65it/s, acc=0.0%, lr=0.000982, total_loss=3.8]
Validation Epoch 11/100: 100%|████████████████████| 16/16 [00:00<00:00, 21.44it/s, total_loss=3.82, val_acc=0.0%]
Train Epoch 12/100: 100%|██████████████| 119/119 [00:08<00:00, 14.23it/s, acc=0.0%, lr=0.000977, total_loss=3.67]
Validation Epoch 12/100: 100%|████████████████████| 16/16 [00:00<00:00, 21.25it/s, total_loss=2.79, val_acc=0.0%]
Train Epoch 13/100: 100%|██████████████| 119/119 [00:08<00:00, 14.52it/s, acc=0.0%, lr=0.000971, total_loss=3.37]
Validation Epoch 13/100: 100%|████████████████████| 16/16 [00:00<00:00, 20.81it/s, total_loss=2.78, val_acc=0.0%]
Train Epoch 14/100: 100%|██████████████| 119/119 [00:08<00:00, 14.26it/s, acc=0.0%, lr=0.000965, total_loss=2.77]
Validation Epoch 14/100: 100%|████████████████████| 16/16 [00:00<00:00, 20.50it/s, total_loss=2.88, val_acc=0.0%]
Train Epoch 15/100: 100%|████████████| 119/119 [00:08<00:00, 13.95it/s, acc=16.51%, lr=0.000959, total_loss=1.77]
Validation Epoch 15/100: 100%|███████████████████| 16/16 [00:00<00:00, 20.41it/s, total_loss=2.99, val_acc=7.94%]
Train Epoch 16/100: 100%|███████████| 119/119 [00:08<00:00, 14.16it/s, acc=98.01%, lr=0.000952, total_loss=0.556]
Validation Epoch 16/100: 100%|█████████████████| 16/16 [00:00<00:00, 19.16it/s, total_loss=0.888, val_acc=91.75%]
Train Epoch 17/100: 100%|███████████| 119/119 [00:08<00:00, 14.03it/s, acc=99.29%, lr=0.000945, total_loss=0.319]
Validation Epoch 17/100: 100%|██████████████████| 16/16 [00:00<00:00, 20.63it/s, total_loss=0.77, val_acc=70.56%]
Train Epoch 18/100: 100%|███████████| 119/119 [00:08<00:00, 14.40it/s, acc=99.25%, lr=0.000936, total_loss=0.234]
Validation Epoch 18/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.23it/s, total_loss=0.221, val_acc=99.19%]
Train Epoch 19/100: 100%|███████████| 119/119 [00:08<00:00, 14.09it/s, acc=98.95%, lr=0.000928, total_loss=0.185]
Validation Epoch 19/100: 100%|███████████████████| 16/16 [00:00<00:00, 20.02it/s, total_loss=3.66, val_acc=1.31%]
Train Epoch 20/100: 100%|███████████| 119/119 [00:08<00:00, 14.20it/s, acc=98.91%, lr=0.000919, total_loss=0.155]
Validation Epoch 20/100: 100%|█████████████████| 16/16 [00:00<00:00, 19.85it/s, total_loss=0.396, val_acc=96.44%]
Train Epoch 21/100: 100%|███████████| 119/119 [00:08<00:00, 13.73it/s, acc=98.85%, lr=0.000909, total_loss=0.129]
Validation Epoch 21/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.29it/s, total_loss=0.207, val_acc=98.19%]
Train Epoch 22/100: 100%|███████████| 119/119 [00:08<00:00, 14.09it/s, acc=99.44%, lr=0.000899, total_loss=0.104]
Validation Epoch 22/100: 100%|███████████████████| 16/16 [00:00<00:00, 20.94it/s, total_loss=3.66, val_acc=28.5%]
Train Epoch 23/100: 100%|████████████| 119/119 [00:08<00:00, 13.86it/s, acc=99.5%, lr=0.000889, total_loss=0.088]
Validation Epoch 23/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.66it/s, total_loss=0.441, val_acc=95.62%]
Train Epoch 24/100: 100%|██████████| 119/119 [00:08<00:00, 13.76it/s, acc=99.81%, lr=0.000878, total_loss=0.0738]
Validation Epoch 24/100: 100%|███████████████████| 16/16 [00:00<00:00, 20.45it/s, total_loss=0.28, val_acc=99.5%]
Train Epoch 25/100: 100%|██████████| 119/119 [00:08<00:00, 13.93it/s, acc=99.61%, lr=0.000867, total_loss=0.0675]
Validation Epoch 25/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.69it/s, total_loss=0.277, val_acc=98.19%]
Train Epoch 26/100: 100%|██████████| 119/119 [00:08<00:00, 13.78it/s, acc=99.61%, lr=0.000855, total_loss=0.0604]
Validation Epoch 26/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.32it/s, total_loss=0.102, val_acc=99.06%]
Train Epoch 27/100: 100%|██████████| 119/119 [00:08<00:00, 13.72it/s, acc=99.42%, lr=0.000843, total_loss=0.0582]
Validation Epoch 27/100: 100%|███████████████| 16/16 [00:00<00:00, 20.27it/s, total_loss=0.00201, val_acc=99.06%]
Train Epoch 28/100: 100%|███████████| 119/119 [00:08<00:00, 14.29it/s, acc=99.72%, lr=0.00083, total_loss=0.0484]
Validation Epoch 28/100: 100%|█████████████████| 16/16 [00:00<00:00, 20.01it/s, total_loss=0.262, val_acc=97.62%]
Train Epoch 29/100: 100%|██████████| 119/119 [00:08<00:00, 13.95it/s, acc=99.13%, lr=0.000817, total_loss=0.0547]
```
#### 自定义训练集
方式一：按照原有数据集方式，将训练数据放入到data目录下，按照label_xx.jpg 方式命名，其中label为标签  
方式二：修改tool/dataloader.py，按照你的数据集方式读取数据

## 更新说明
#### 2023.9.15更新: 提交项目 

# 实现原理


# 参考文档
https://github.com/DayBreak-u/chineseocr_lite/tree/master  
https://github.com/meijieru/crnn.pytorch  