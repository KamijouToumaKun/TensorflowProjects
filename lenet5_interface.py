import tensorflow as tf

# 每一层的节点数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
# 第一层卷积层的尺寸和深度；尺寸一般不超过5
CONV1_SIZE = 5
CONV1_DEEP = 32
# 第二层卷积层的尺寸和深度；深度一般逐层翻倍
CONV2_SIZE = 5
CONV2_DEEP = 64
# 池化层一般大小和步长同为2或3
# 全连接层的节点个数
INPUT_NODE = None #待定
FC_NODE = 512
OUTPUT_NODE = 10

def get_weight_variable(shape, regularizer):
    # 获取权重，如果没有的话则新建
    w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 自定义一个集合loss，以后用于计算正则项
    if regularizer != None:
        tf.add_to_collection('loss', regularizer(w))
    return w

# 这个函数只会调用一次；调用多次的话则因为没有设置reuse而报错
# 多加了一个train参数，用于dropout
# dropout只在训练时使用，用于提升模型可靠性和防止过拟合
def interface(input_tensor, train, regularizer):
    # regularizer参数的作用是给在本variable_scope下创建的weights加上正则项
    # 这样我们就可以不同variable_scope下的参数加不同的正则项了.
    # 28*28*1 -> 28*28*32（大小不变是因为有padding）
    with tf.variable_scope('conv1'):
        w = tf.get_variable("w", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 还是需要添加偏置项
        b = tf.get_variable("b", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 过滤器移动的步长为1，全0填充；卷积的时候不需要偏置项
        conv = tf.nn.conv2d(input_tensor, w, strides=[1,1,1,1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    # 28*28*32 -> 14*14*32
    with tf.variable_scope('pool1'):
        # 采用最大池化，步长为2
        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # 14*14*32 -> 14*14*64
    with tf.variable_scope('conv2'):
        w = tf.get_variable("w", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # 过滤器移动的步长为1，全0填充；卷积的时候不需要偏置项
        conv = tf.nn.conv2d(pool, w, strides=[1,1,1,1], padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, b))
    # 14*14*64 -> 7*7*64
    with tf.variable_scope('pool2'):
        # 采用最大池化，大小和步长为2
        pool = tf.nn.max_pool(relu, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    # 输入是把7*7*64的三维数组平铺成一维；BATCH_SIZE那一维仍旧保留
    pool_shape = pool.get_shape().as_list()
    INPUT_NODE = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool, [pool_shape[0], INPUT_NODE]) 
    with tf.variable_scope('fc1'):
        w = get_weight_variable([INPUT_NODE, FC_NODE], regularizer)
        # b不被记录在正则项中
        b = tf.get_variable("b", [FC_NODE], initializer=tf.constant_initializer(0.0))
        # 这里可以让fc1和scope的fc1同名；注意fc1和fc2要区分开
        fc1 = tf.nn.relu(tf.matmul(reshaped, w) + b)
        # 加入dropout
        # 在训练的时候，我们只需要按一定的概率（retaining probability）p 来对weight layer 的参数进行随机采样
        # 将这个子网络作为此次更新的目标网络。
        # 可以想象，如果整个网络有n个参数，那么我们可用的子网络个数为 2^n 。 
        # 并且，当n很大时，每次迭代更新 使用的子网络基本上不会重复，从而避免了某一个网络被过分的拟合到训练集上。
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('fc2'):
        w = get_weight_variable([FC_NODE, OUTPUT_NODE], regularizer)
        b = tf.get_variable("b", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        fc2 = tf.matmul(fc1, w) + b
    return fc2