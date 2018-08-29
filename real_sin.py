import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# 截取出训练序列所用的窗口大小
TIMESTEPS = 10
# 产生数据：对于一个序列，用大小TIMESTEP的窗口去截取seq序列，用前面TIMESTEP-1个点预测最后一个点
def generate_data(seq):
    X = []
    Y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        Y.append([seq[i+TIMESTEPS]])
    # return np.array(X), np.array(Y) 不写类型好像也没有什么问题
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
# 搭建RNN-LSTM
# 训练和测试时传入的X和Y的格式不一样：训练时有BATCH_SIZE而测试时没有
# 所以需要在每次调用时重新设置dynamic_rnn 但是在测试处调用interface时一定要注明重用
def interface(X, Y, is_training):
    # 各层的定义
    HIDDEN_SIZE = 30 # RNN-LSTM每层的节点数
    NUM_LAYERS = 2 #RNN-LSTM层数
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    # dynamic_rnn函数要求必须指明类型
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=np.float32)
    # 三个维度分别是[batch_size, time, layer]，这里在第二维只取最后的一个时刻
    output = outputs[:,-1,:]
    # 最后再连接一个全连接层，这里直接使用封装函数；不使用激励函数
    predict_op = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    if not is_training:
        return predict_op, None, None

    # 损失函数 这里是回归问题而不是分类问题，所以使用平方差损失函数
    loss = tf.losses.mean_squared_error(predictions=predict_op,labels=Y)
    # 梯度下降 采用封装好的更高级的模型优化器
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return predict_op, loss, train_op

def main(epoch=10000):
    TRAINING_EXAMPLES = 10000
    TESTING_EXAMPLES = 1000
    # 采样间隔
    SAMPLE_GAP = 0.01
    # 每次训练样本数
    BATCH_SIZE = 32

    test_start = (TRAINING_EXAMPLES+TIMESTEPS) * SAMPLE_GAP
    test_end = test_start + (TRAINING_EXAMPLES+TIMESTEPS) * SAMPLE_GAP
    # dynamic_rnn处定义了dtype，这里必须与其保持一致
    train_data,train_label = generate_data(np.sin(
        np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
    test_data,test_label = generate_data(np.sin(
        np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
    # 因为dynamic_rnn的参数要求特殊，这里不使用placeholder进行定义X & Y
    # 以致于一系列变量的定义都有很大区别
    # 在这里就定义了X的打乱和BATCH_SIZE
    ds = tf.data.Dataset.from_tensor_slices((train_data,train_label))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, Y = ds.make_one_shot_iterator().get_next()
    with tf.variable_scope('model'):
        predict_op, loss, train_op = interface(X, Y, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(epoch):
            # 打乱训练集以及分BATCH_SIZE已经在train_op中定义了，这里不需要
            # 同时也不要写feed_dict了，而直接像下面这样更新并输出loss
            _, l = sess.run([train_op, loss])
            if i % 100 == 0:
               print(i, l)
        # 测试集
        ds = tf.data.Dataset.from_tensor_slices((test_data,test_label))
        ds = ds.batch(1)
        X, Y = ds.make_one_shot_iterator().get_next()
        # 测试时不需要X的打乱和BATCH_SIZE，需要另外设置
        with tf.variable_scope('model', reuse=True):
            predict_op, _, _ = interface(X, None, False)
        predictions = []
        labels = []
        for i in range(TESTING_EXAMPLES):
            p, l = sess.run([predict_op, Y])
            predictions.append(p)
            labels.append(l)
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(((predictions-labels) ** 2).mean(axis=0))
        print(rmse)
        # 作图将模拟结果可视化
        plt.figure()
        plt.plot(predictions, label='predictions')
        plt.plot(labels, label='real-sin')
        plt.legend()
        plt.show()

if __name__=='__main__':
    main()