import numpy as np
import tensorflow as tf
# 提取特征
def get_feature(i, num_digits):
    # return np.array([i >> d & 1 for d in range(num_digits)])
    # 用二进制的效果不佳，改用十进制
    return np.array([i // (10**d) % 10 / 10 for d in range(num_digits)])
    # 把参数归一化到0~1才有效果；注意区分：/表示除法、//表示向下取整的除法
# 得到label：0~8个0，一共九个类，用one-hot方式表示
def get_label(i, num_digits):
    sum = 0
    for t in range(num_digits):
        # 要把i给整数化，否则会有误差
        digit = round(i % 10)
        if digit == 0: sum += 1
        elif digit == 6: sum += 1
        elif digit == 8: sum += 2
        elif digit == 9: sum += 1
        i //= 10
    res = [0 for t in range(9)]
    res[sum] = 1
    return np.array(res)
def generate_train_data(NUM_DIGITS, _range):
    _data = np.array([get_feature(i,NUM_DIGITS) for i in _range])
    _label = np.array([get_label(i,NUM_DIGITS) for i in _range])
    return _data,_label
def predict2word(num,prediction):
    return prediction
def main(epoch=10000):
    NUM_DIGITS = 4
    train_data,train_label = generate_train_data(NUM_DIGITS, range(101,10001,1))
    # 每次训练样本数
    BATCH_SIZE = 256
    # 输入层和输出层
    # 定义每一层的神经元数量，不能太少；第一层是特征数，最后一层是类别数
    NUM_NODE = [NUM_DIGITS, 128, 32, 9]
    X = tf.placeholder('float32',[None, NUM_NODE[0]])
    Y = tf.placeholder('float32',[None, NUM_NODE[3]])
    # 用正态分布的随机值进行初始化
    w1 = tf.Variable(tf.random_normal([NUM_NODE[0], NUM_NODE[1]]))
    b1 = tf.Variable(tf.random_normal([NUM_NODE[1]]))
    w2 = tf.Variable(tf.random_normal([NUM_NODE[1], NUM_NODE[2]]))
    b2 = tf.Variable(tf.random_normal([NUM_NODE[2]]))
    w3 = tf.Variable(tf.random_normal([NUM_NODE[2], NUM_NODE[3]]))
    b3 = tf.Variable(tf.random_normal([NUM_NODE[3]]))
    # 各层的定义；均采用全连接
    fc1 = tf.nn.relu(tf.matmul(X,w1) + b1)
    fc2 = tf.nn.relu(tf.matmul(fc1,w2) + b2)
    out = tf.matmul(fc2,w3) + b3
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))
    # 尝试改用，类别编号和真实类别编号之间的差距；因为答案只有一个正确数字，可以改用sparse加速
    # 但是这样也没有快多少，而且训练的时候准确率的抖动很大
    # 解决方法：0）滑动平均；1）减小学习率（退火）；2）使用一个自适应的batch大小。
    # 而且测试集的准确率明显降低了？？？就算把神经元从128增到256个也还是不高
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=tf.argmax(Y,1)))
    # 梯度下降
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    predict_op = tf.argmax(out, 1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(epoch):
            # 打乱训练集
            rand_indice = np.random.permutation(range(len(train_data)))
            train_data,train_label = train_data[rand_indice],train_label[rand_indice]
            # 每次取BATCH_SIZE个样本进行训练
            # 如果不打乱，训练效果会大大变差！训练4000轮，训练集的正确率从0.998变成只有0.790
            for start in range(0,len(train_data)-1,BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_op,feed_dict={X:train_data[start:end],Y:train_label[start:end]})
            # 输出准确率
            print(i, np.mean(np.argmax(train_label, axis=1) == sess.run(predict_op, feed_dict={X: train_data, Y: train_label})))
        # 测试集
        _range = np.arange(1, 101)
        test_data = np.transpose(get_feature(_range, NUM_DIGITS))
        test_label = sess.run(predict_op, feed_dict={X: test_data})
        output = np.vectorize(predict2word)(_range, test_label)
        print(output)
if __name__=='__main__':
    main(4000)