#-*- coding:utf-8 -*-
import tensorflow as tf

# 每一层的节点数
NUM_NODE = [28*28, 500, 2]

def get_weight_variable(shape, regularizer):
    # 获取权重，如果没有的话则新建
    w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 自定义一个集合loss，以后用于计算正则项
    if regularizer != None:
        tf.add_to_collection('loss', regularizer(w))
    return w

# 计算三层神经网络（一个隐藏层）的结果
# 这个函数只会调用一次；调用多次的话则因为没有设置reuse而报错
def interface(input_tensor, regularizer):
    # regularizer参数的作用是给在本variable_scope下创建的weights加上正则项
    # 这样我们就可以不同variable_scope下的参数加不同的正则项了.
    with tf.variable_scope('fc1'):
        w = get_weight_variable([NUM_NODE[0], NUM_NODE[1]], regularizer)
        # b不被记录在正则项中
        b = tf.get_variable("b", [NUM_NODE[1]], initializer=tf.constant_initializer(0.0))
        # 这里可以让fc1和scope的fc1同名；注意fc1和fc2要区分开
        fc1 = tf.nn.relu(tf.matmul(input_tensor, w) + b)
    with tf.variable_scope('fc2'):
        w = get_weight_variable([NUM_NODE[1], NUM_NODE[2]], regularizer)
        b = tf.get_variable("b", [NUM_NODE[2]], initializer=tf.constant_initializer(0.0))
        fc2 = tf.matmul(fc1, w) + b
    return fc2