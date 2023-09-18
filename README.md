
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
识别结果：aqqh，推理耗时：8.0ms
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