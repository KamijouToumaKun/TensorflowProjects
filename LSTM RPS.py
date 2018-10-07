# -*- coding: utf-8 -*- 

import numpy as np
import tensorflow as tf

import matplotlib as mpl
# mpl.use('Agg') 加了这一行则图片不会显示出来
from matplotlib import pyplot as plt

# 1. 设置神经网络的参数
HIDDEN_SIZE = 30 # 隐藏节点个数
NUM_LAYERS = 2 # 层数

TIMESTEPS = 10 # 训练序列长度。用前面TIMESTEPS个数据预测当前点的值
TRAINING_STEPS = 2000 # EPOCH数；随机因素太大，多迭代几次并没有用
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 1 # 采样间隔

# 2. 定义生成正弦数据的函数
def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 3. 定义lstm模型
def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

    # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    # 将predictions和labels调整统一的shape
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op

def train(sess, train_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print "Train step: ", str(i), ", loss: ", str(l)

def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        prediction, _, _, = lstm_model(X, [0.0], False)

    predictions = []
    labels = []
    sum = 0
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(round(p))
        labels.append(l)
        if round(p) == l:
            sum += 1

    print 1.0*sum/TESTING_EXAMPLES

NUM_CUSTOM = 5
LEN_PER_CUSTOM = 4
# 假设出拳有 NUM_CUSTOM 种套路，一般来说也就5种套路吧，10种太多了
custom_index = [[1,2,3,3],[1,1,2,3],[2,1,1,1],[3,3,2,3],[3,1,2,3]]
# custom_index = [[1,2,3,3],[1,1,3,3],[2,2,2,2],[3,3,1,2],[1,2,2,3]]
# custom_index.extend([[1,2,3,3],[1,1,2,3],[2,1,1,1],[3,3,2,3],[3,1,2,3]])
# 生成 TRAINING_EXAMPLES 个1 ~ NUM_CUSTOM之间的整数
# 表示对方出拳是随机的，但一定是在这些套路中挑一个
indice = np.random.random_integers(NUM_CUSTOM, size=TRAINING_EXAMPLES)

def get_custom(index):
    return custom_index[indice[int(index)/LEN_PER_CUSTOM]-1][int(index)%LEN_PER_CUSTOM]

if __name__ == '__main__':
    get_custom_seq = np.vectorize(get_custom)
    test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    train_X, train_y = generate_data(get_custom_seq(np.linspace(
        0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_X, test_y = generate_data(get_custom_seq(np.linspace(
        test_start, test_end, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

    with tf.Session() as sess:
        train(sess, train_X, train_y)
        run_eval(sess, test_X, test_y)
