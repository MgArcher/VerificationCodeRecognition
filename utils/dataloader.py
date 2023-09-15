

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


def preprocess_input(x):
    x /= 255.0
    return x


def get_paths(path):
    paths = glob.glob(f'{path}/**.jpg') + glob.glob(f'{path}/**.png')
    if not paths:
        paths = glob.glob(f'{path}/*/**.jpg') + glob.glob(f'{path}/*/**.png')
    if not paths:
        paths = glob.glob(f'{path}/*/*/**.jpg') + glob.glob(f'{path}/*/*/**.png')
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
    # widht = int(w * (opt.HEIGHT / h))
    # out = out.resize((widht, opt.HEIGHT), Image.ANTIALIAS)
    out = out.resize((w, h), Image.ANTIALIAS)
    if nc == 1:
        out = out.convert('L')
    return out


class CaptchaDataset(Dataset):
    def __init__(self, dataset_path, input_shape,charactes_keys, nc):
        lines, labels = load_dataset(dataset_path)
        self.input_shape    = input_shape
        self.train_lines    = lines
        self.train_labels   = labels
        self.charactes_keys = charactes_keys
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
