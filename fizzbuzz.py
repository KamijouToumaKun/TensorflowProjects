import numpy as np
import tensorflow as tf

# 当然，现在让我们来讨论一下模型，我正在想一个简单的只有一个隐藏层的感知器。

# 首先我们需要把数字转为向量，最简单的方法是把数字转换为二进制表示。
def binary_encode(i, num_digits):
    return np.array([i>>d&1 for d in range(num_digits)])
# 输出应该用one-hot编码表示”fizz buzz”
def fizz_buzz_encode(i):
    # 要把i给整数化，否则会有误差
    i = round(i)
    if   i % 15 == 0: return np.array([0, 0, 0, 1]) # FizzBuzz
    elif i % 5  == 0: return np.array([0, 0, 1, 0]) # Buzz
    elif i % 3  == 0: return np.array([0, 1, 0, 0]) # Fizz
    else:             return np.array([1, 0, 0, 0])
# 这个网络只有两层深，一个隐藏层和一个输出层。下面，让我们使用随机数初始化“神经元”的权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
# 现在我们可以定义模型了，就像我前面说的，一个隐藏层。激活函数用什么呢，我不知道，就用ReLU吧
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)
# 现在predict_op输出的值是0-3，还要转换为”fizz buzz”输出
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

with tf.Session() as sess:
    # 基本的准备工作已经完成了。现在我们需要生成一个训练数据
    # 我们不用1到100训练，为了增加难度，我们使用100-1024训练
    NUM_DIGITS = 10
    trX = np.array([binary_encode(i,NUM_DIGITS) for i in range(101, 2**NUM_DIGITS)])
    trY = np.array([fizz_buzz_encode(i) for i in range(101, 2**NUM_DIGITS)])
    # 定义输入和输出
    X = tf.placeholder("float", [None,NUM_DIGITS])
    Y = tf.placeholder("float", [None,4])
    # 隐藏层的神经元先设置为NUM_HIDDEN = 100
    NUM_HIDDEN = 100
    w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
    w_o = init_weights([NUM_HIDDEN, 4])
    # 我们可以使用softmax cross-entrop做为loss函数，并且试图最小化它。
    py_x = model(X, w_h, w_o) 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
    # 最后还要取概率最大的预测做为结果
    predict_op = tf.argmax(py_x, 1)
    tf.global_variables_initializer().run()
    # 为了保险就训练10000次。我们的训练数据是生成的序列，最好在每个epoch随机打乱一下
    for epoch in range(10000):
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]
        # 每次取128个样本进行训练
        BATCH_SIZE = 128
        # 训练
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        # 查看准确率
        print(epoch, np.mean(np.argmax(trY,axis=1) ==
            sess.run(predict_op, feed_dict={X:trX, Y:trY})))
    # 实际测试
    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X:teX})
    output = np.vectorize(fizz_buzz)(numbers,teY)
    print(output)