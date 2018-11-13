#-*- coding:utf-8 -*-
import os 
import tensorflow as tf 
from PIL import Image
import numpy as np

import mnist_interface

class mydataset(object):
    """docstring for mydataset"""
    def __init__(self, cwds, classes):
        super(mydataset, self).__init__()
        self.img = []
        self.label = []
        self.count = 0
        self.load_data(cwds, classes)

    def load_data(self, cwds, classes):
        H = W = int(np.sqrt(mnist_interface.NUM_NODE[0]))
        for cwd in cwds:
            for index, name in enumerate(classes):
                class_path = cwd + '/' + name + '/'
                for img_name in os.listdir(class_path): 
                    if img_name == '.DS_Store':
                        continue
                    img_path = class_path + img_name # 每一个图片的地址
                    img = Image.open(img_path)
                    img = img.resize((H,W)) # 大小规范化
                    img = img.convert('L') # 转成灰度图
                    img_raw = np.array(img) / 256 # 归一化
                    img_raw = img_raw.reshape(-1).tolist() # 转成一维H*W
                    self.img.append(img_raw)
                    score = [0 for _ in range(mnist_interface.NUM_NODE[2])]
                    score[index] = 1
                    self.label.append(score) # C个得分
        self.img = np.array(self.img)
        self.label = np.array(self.label)
        # array(N, H*W), array(N, C)

    def next_batch(self, BATCH_SIZE):
        self.count += BATCH_SIZE
        if self.count >= len(self.img): # 到头则重新打乱
            permutation = np.random.permutation(len(self.img))
            self.img = self.img[permutation]
            self.label = self.label[permutation]
            self.count = BATCH_SIZE
        return self.img[self.count-BATCH_SIZE:self.count],\
            self.label[self.count-BATCH_SIZE:self.count]
