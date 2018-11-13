#-*- coding:utf-8 -*-
import time

import tensorflow as tf
from mydataset import mydataset

import mydataset_interface
import mydataset_train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mydata):
    # 重用graph
    with tf.Graph().as_default() as g:  
        X = tf.placeholder(tf.float32, [None, mydataset_interface.NUM_NODE[0]], name='x-input')
        Y = tf.placeholder(tf.float32, [None, mydataset_interface.NUM_NODE[2]], name='y-input')
        # 不使用滑动平均类的输出；测试的时候不再使用regularizer、加入loss集合了
        avg_out = mydataset_interface.interface(X, None)
        # 计算准确率
        # 训练时最小化损失函数用的是out，但是计算准确率（无论是验证还是测试）看的是avg_out
        correct_prediction = tf.equal(tf.argmax(avg_out, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重用的方式来加载模型？？？
        avg_class = tf.train.ExponentialMovingAverage(mydataset_train.MOVING_AVERAGE_DECAY)
        avg_class_to_restore = avg_class.variables_to_restore()
        saver = tf.train.Saver(avg_class_to_restore)
        # 每隔一定时间测试一次最新的模型的准确率
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mydataset_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型加载时迭代的轮数
                    epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print(epoch, sess.run(accuracy, feed_dict={X:mydata.img, Y:mydata.label}))
                else:
                    print('No checkpoint file found')
                    return
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mytestdata = mydataset(['1', '2'], ['画', '真人'])
    evaluate(mytestdata)

if __name__ == '__main__':
    tf.app.run()    