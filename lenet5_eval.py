import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import lenet5_interface
import lenet5_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    # 重用graph
    with tf.Graph().as_default() as g:
        # 需要reshape
        reshaped_xs = np.reshape(mnist.validation.images, (-1, lenet5_interface.IMAGE_SIZE, lenet5_interface.IMAGE_SIZE, lenet5_interface.NUM_CHANNELS))
        input_shape = reshaped_xs.shape
        X = tf.placeholder(tf.float32, [input_shape[0], lenet5_interface.IMAGE_SIZE, lenet5_interface.IMAGE_SIZE, lenet5_interface.NUM_CHANNELS], name='x-input')
        Y = tf.placeholder(tf.float32, [None, lenet5_interface.OUTPUT_NODE], name='y-input')
        # 使用滑动平均类的输出；测试的时候不再使用regularizer、加入loss集合了
        avg_out = lenet5_interface.interface(X, False, None)
        # 计算准确率
        # 训练时最小化损失函数用的是out，但是计算准确率（无论是验证还是测试）看的是avg_out
        correct_prediction = tf.equal(tf.argmax(avg_out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重用的方式来加载模型？？？
        avg_class = tf.train.ExponentialMovingAverage(lenet5_train.MOVING_AVERAGE_DECAY)
        avg_class_to_restore = avg_class.variables_to_restore()
        saver = tf.train.Saver(avg_class_to_restore)
        # 每隔一定时间测试一次最新的模型的准确率
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型加载时迭代的轮数
                    epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 需要代入reshape的xs
                    print(epoch, sess.run(accuracy, feed_dict={X:reshaped_xs, Y:mnist.validation.labels}))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()                