import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_interface
# 一次训练的样本数；数量越少则越接近以及梯度下降，数量越多则越接近梯度下降
BATCH_SIZE = 100
# 学习率和学习率的衰减率
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
# 正则化项的系数
REGULARIZATION_RATE = 1e-4
TRAINING_EPOCH = 20000
# 滑动平均衰减率，指数衰减
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model'
MODEL_NAME = 'model.ckpt'

def train(mnist):
    X = tf.placeholder(tf.float32, [None, mnist_interface.NUM_NODE[0]], name='x-input')
    Y = tf.placeholder(tf.float32, [None, mnist_interface.NUM_NODE[2]], name='y-input')

    # l2正则化损失函数；这里因为在out & avg_out要用到，所以提前定义
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 不使用滑动平均类的输出
    out = mnist_interface.interface(X, regularizer)
    # 在计算图上也声明一个计算轮数的变量；标明它是不可训练的，就不会被初始化了
    epoch = tf.Variable(0, trainable=False)
    avg_class = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, epoch)
    avg_class_op = avg_class.apply(tf.trainable_variables())
    # 使用滑动平均类的输出，现在放到mnist_eval中定义：带regularizer参数的interface只调用一次
    # avg_out = mnist_interface.interface(X, regularizer)

    # 损失函数
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))
    # 这里改用，类别编号和真实类别编号之间的差距（这样好吗？？？）
    # 因为答案只有一个正确数字，可以改用sparse加速
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=tf.argmax(Y,1)))
    # 计算模型的正则化损失；从loss集合中获取
    regularization = tf.add_n(tf.get_collection('loss'))
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，将会逐渐递减
        epoch, # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE, # 过完所有的训练数据需要的迭代轮数
        LEARNING_RATE_DECAY) # 衰减率
    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=epoch)
    # 之后还要更新滑动平均值？？？
    with tf.control_dependencies([train_step, avg_class_op]):
        train_op = tf.no_op(name='train')

    # 初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_EPOCH):
            # 训练集
            if i % 1000 == 0:
                print(i)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=epoch)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={X:xs, Y:ys})

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()