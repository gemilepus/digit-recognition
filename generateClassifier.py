#!/usr/bin/python
# coding=utf-8

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
# import collection
import os
import struct


def load_mnist(path):
    # 讀取資料函數
    # Load MNIST data from path
    labels_path = 'train-labels.idx1-ubyte'
    images_path = 'train-images.idx3-ubyte'

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


features, labels = load_mnist("./")
print('Rows: %d, columns: %d' % (features.shape[0], labels.shape[0]))

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)),     # hog 特徵
             orientations=9,
             pixels_per_cell=(14, 14),
             cells_per_block=(1, 1),
             visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()                                # 定義分類器
clf.fit(hog_features, labels)                    # 訓練
joblib.dump(clf, "digits_cls.pkl", compress=3)   # 模型保存

# 壓縮：0到9的整數可選
# 壓縮層級：0沒有壓縮。越高意味著更多的壓縮，而且讀取和寫入越慢。使用3的值通常是一個很好的折衷。