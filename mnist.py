import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 每一层的节点数
NUM_NODE = [28*28, 500, 10]
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

# 计算三层神经网络（一个隐藏层）的结果，可能需要使用到滑动平均类
def interface(input_tensor, avg_class, w1, b1, w2, b2):
    if avg_class == None:
        fc = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
        return tf.matmul(fc, w2) + b2
    else:
        avg_w1 = avg_class.average(w1)
        avg_b1 = avg_class.average(b1)
        avg_w2 = avg_class.average(w2)
        avg_b2 = avg_class.average(b2)
        fc = tf.nn.relu(tf.matmul(input_tensor, avg_w1) + avg_b1)
        return tf.matmul(fc, avg_w2) + avg_b2

def train(mnist):
    X = tf.placeholder(tf.float32, [None, NUM_NODE[0]], name='x-input')
    Y = tf.placeholder(tf.float32, [None, NUM_NODE[2]], name='y-input')
    w1 = tf.Variable(tf.truncated_normal([NUM_NODE[0], NUM_NODE[1]], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[NUM_NODE[1]]))
    # b1 = tf.Variable(tf.truncated_normal([NUM_NODE[1]], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([NUM_NODE[1], NUM_NODE[2]], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[NUM_NODE[2]]))
    # b2 = tf.Variable(tf.truncated_normal([NUM_NODE[2]], stddev=0.1))
    # 不使用滑动平均类的输出
    out = interface(X, None, w1, b1, w2, b2)
    # 在计算图上也声明一个计算轮数的变量；标明它是不可训练的，就不会被初始化了
    epoch = tf.Variable(0, trainable=False)
    avg_class = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, epoch)
    avg_class_op = avg_class.apply(tf.trainable_variables())
    # 使用滑动平均类的输出
    avg_out = interface(X, avg_class, w1, b1, w2, b2)

    # 损失函数
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))
    # 这里改用，类别编号和真实类别编号之间的差距（这样好吗？？？）
    # 因为答案只有一个正确数字，可以改用sparse加速
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=tf.argmax(Y,1)))
    # l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失；一般只计算权重w而不计算偏置b
    regularization = regularizer(w1) + regularizer(w2)
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

    # 计算准确率
    # 训练时最小化损失函数用的是out，但是计算准确率（无论是验证还是测试）看的是avg_out
    correct_prediction = tf.equal(tf.argmax(avg_out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_EPOCH):
            # 训练集
            if i % 1000 == 0:
                # 输出验证集上的准确率
                # 可以写成这样的形式
                # print(i, np.mean(np.argmax(train_label, axis=1) == sess.run(predict_op, feed_dict={X: train_data, Y: train_label})))
                # 即mean(argmax(train) == run(argmax(out), feed_dict=train))
                # 在这里写成这样的形式
                print(i, sess.run(accuracy, feed_dict={X:mnist.validation.images, Y:mnist.validation.labels}))
                # 即run(mean(argmax(out) == argmax(Y)), feed_dict=train)
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={X:xs, Y:ys})
        # 测试集准确率
        print(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()