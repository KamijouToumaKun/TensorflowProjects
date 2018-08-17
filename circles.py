import numpy as np
import tensorflow as tf
def binary_encode(i, num_digits):
    # return np.array([i >> d & 1 for d in range(num_digits)])
    # 用二进制的效果不佳，改用十进制
    return np.array([i // (10**d) % 10 / 10 for d in range(num_digits)])
    # 把参数归一化到0~1才有效果；注意区分：/表示除法、//表示向下取整的除法
def circle_encode(i, num_digits):
    sum = 0
    for t in range(num_digits):
        # 要把i给整数化，否则会有误差
        digit = round(i % 10)
        if digit == 0: sum += 1
        elif digit == 6: sum += 1
        elif digit == 8: sum += 2
        elif digit == 9: sum += 1
        i //= 10
    # 0~8个0，一共九个类，用one-hot方式表示
    res = [0 for t in range(9)]
    res[sum] = 1
    return np.array(res)
def generate_train_data(NUM_DIGITS, _range):
    _data = np.array([binary_encode(i,NUM_DIGITS) for i in _range])
    _label = np.array([circle_encode(i,NUM_DIGITS) for i in _range])
    return _data,_label
def predict2word(num,prediction):
    return prediction
def main(epoch=10000):
    NUM_DIGITS = 4
    train_data,train_label = generate_train_data(NUM_DIGITS, range(101,10001,1))
    # 输入层和输出层
    # 定义每一层的神经元数量，不能太少
    NUM_NEURON = [0, 128, 32, 9]
    X = tf.placeholder('float32',[None, NUM_DIGITS])
    Y = tf.placeholder('float32',[None, NUM_NEURON[3]])
    # 用正态分布的随机值进行初始化
    w1 = tf.Variable(tf.random_normal([NUM_DIGITS, NUM_NEURON[1]]))
    b1 = tf.Variable(tf.random_normal([NUM_NEURON[1]]))
    w2 = tf.Variable(tf.random_normal([NUM_NEURON[1], NUM_NEURON[2]]))
    b2 = tf.Variable(tf.random_normal([NUM_NEURON[2]]))
    w3 = tf.Variable(tf.random_normal([NUM_NEURON[2], NUM_NEURON[3]]))
    b3 = tf.Variable(tf.random_normal([NUM_NEURON[3]]))
    # 全连接
    fc1 = tf.nn.relu(tf.matmul(X,w1) + b1)
    fc2 = tf.nn.relu(tf.matmul(fc1,w2) + b2)
    out = tf.matmul(fc2,w3) + b3
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=Y))
    # 输出概率最高的作为结果
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    predict_op = tf.argmax(out, 1)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(epoch):
            # 打乱训练集
            rand_indice = np.random.permutation(range(len(train_data)))
            train_data,train_label = train_data[rand_indice],train_label[rand_indice]
            # 每次训练样本数
            batch_size = 256
            for start in range(0,len(train_data)-1,batch_size):
                end = start + batch_size
                sess.run(train_op,feed_dict={X:train_data[start:end],Y:train_label[start:end]})
            # 输出准确率
            print(i, np.mean(np.argmax(train_label, axis=1) == sess.run(predict_op, feed_dict={X: train_data, Y: train_label})))
        # 测试集
        _range = np.arange(1, 101)
        test_data = np.transpose(binary_encode(_range, NUM_DIGITS))
        test_label = sess.run(predict_op, feed_dict={X: test_data})
        output = np.vectorize(predict2word)(_range, test_label)
        print(output)
if __name__=='__main__':
    main(4000)