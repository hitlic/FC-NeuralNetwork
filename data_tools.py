# coding: utf-8
# author: LiuChen

import random
import struct
from mnist import MNIST
import numpy as np

def load_iris_data(path):
    """
    从文件中加载iris数据
    数据的形式为：
    [
        ([x11,x12,...,x1d],[0, 1, 0]),
        ([x21,x22,...,x2d],[0, 0, 1]),
        ...
    ]
    """
    ds = []
    f = open(path)  # 打开文件
    for line in f.readlines():  # 遍历每一行
        items = line.strip().split(',')  # 去掉行首尾的空格，并根据“,”切分每行
        if items[-1] == 'Iris-setosa':  # 判断标记 y 的类型
            y = [1, 0, 0]
        elif items[-1] == 'Iris-versicolor':
            y = [0, 1, 0]
        else:
            y = [0, 0, 1]
        del items[-1]
        x = [float(e) for e in items]  # 将 x 中数据转为整数
        ds.append((x, y))
    random.shuffle(ds)
    return ds


def one_hot(x, dim=10):
    """
    根据数字序列生成One-hot序列
    """
    num = len(x)
    one_hot = np.zeros((num, dim))
    one_hot[np.arange(num), x] = 1
    return one_hot
