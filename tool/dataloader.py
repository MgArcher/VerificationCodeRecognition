

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


def preprocess_input(x):
    x /= 255.0
    return x


def get_paths(path):
    def get_subfolders(folder):
        yield folder
        for root, dirs, files in os.walk(folder):
            for dir in dirs:
                yield os.path.join(root, dir)
    paths = []
    for subfolders in get_subfolders(path):
        paths += glob.glob(f'{subfolders}/**.jpg') + glob.glob(f'{subfolders}/**.png')

    return paths


def load_dataset(dataset_path):
    paths = get_paths(dataset_path)
    lines = []
    labels = []
    for file in paths:
        # print(file)
        name = os.path.basename(file)
        name = os.path.splitext(name)[0]
        # print(name)
        if '_' in name:
            random_str = name.split('_')[0]
        else:
            random_str = name
        random_str = random_str.replace('@', '').replace(' ', '')
        random_str = random_str.lower()
        if not random_str:
            continue
        lines.append(file)
        labels.append(random_str)
    return lines, labels


def ratio_dataloader(lines, labels, train_ratio, batchSize):
    num_train = int(len(lines) * (1 - train_ratio))
    if num_train < batchSize:
        print("数据集不足以划分训练集和验证集，训练集验证集返回相同的")
        return lines, labels, lines, labels
    train_lines = lines[num_train:]
    train_labels = labels[num_train:]
    val_lines = lines[:num_train]
    val_labels = labels[:num_train]
    return train_lines, train_labels, val_lines, val_labels


def get_charactes_keys(path):
    charactes_keys = ""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            charactes_keys += line
    return charactes_keys


def open_image(file, input_shape, nc):
    out = Image.open(file)
    # 改变大小 并保证其不失真
    out = out.convert('RGB')
    h, w = input_shape
    out = out.resize((w, h), 1)
    if nc == 1:
        out = out.convert('L')
    return out


class CaptchaDataset(Dataset):
    def __init__(self, dataset_path, input_shape, nc=3):
        if isinstance(dataset_path, tuple) or isinstance(dataset_path, list):
            lines, labels = dataset_path
        else:
            lines, labels = load_dataset(dataset_path)
        self.input_shape    = input_shape
        self.train_lines    = lines
        self.train_labels   = labels
        self.nc = nc

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        lines = self.train_lines[index]
        labels = self.train_labels[index]
        image = open_image(lines, self.input_shape, self.nc)
        image = preprocess_input(np.array(image).astype(np.float32))
        image = np.transpose(image, [2, 0, 1])
        return image, labels




