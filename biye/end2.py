import tensorflow as tf
from numpy.random import RandomState
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import *   # 导入numpy的库函数

isTrain = True
# 会话保存地址
model_path = "D:\sample\model.ckpt"
# 加载 matlab 数据文件
train_data = sio.loadmat('D:\\MATLAB Workspace\\swpuProject\\data_31.mat')
check_data = sio.loadmat('D:\\MATLAB Workspace\\swpuProject\\check100a.mat')

#设置相关的参数
INPUT_NODE = 100        # 输入层节点数
OUTPUT_NODE = 4         # 输出层节点数
LAYER1_NODE = 100       # 第一隐藏层节点数
LAYER2_NODE = 50
BATCH_SIZE = 100        # 训练数据批次大小
LEARNING_RATE_BASE = 0.0001     # 基础学习率
LEARNING_RATE_DECAY = 0.98      # 学习率衰减率
TRAINING_STEPS = 500            # 训练轮数
MOVING_AVERAGE_DECAY = 0.99

#分别获取训练数据的输入和输出，作为神经网络的参数
inputMat = train_data['inputMat']
outputMat = train_data['outputMat']

check_input = check_data['inputMat']
check_output = check_data['outputMat']

# 获取输入数据的维度
m, n = inputMat.shape
# 计算批次总数
n_batch = m // BATCH_SIZE

# 在内存上定义两块空间
x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')



#搭建单隐藏层神经网络模型
w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.zeros([LAYER1_NODE])+0.1)
a = tf.nn.sigmoid(tf.matmul(x, w1)+biases1)
# a = tf.nn.relu(tf.matmul(x, w1)+biases1)
# a = tf.nn.tanh(tf.matmul(x, w1)+biases1)

w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.zeros([OUTPUT_NODE])+0.1)
y = tf.nn.sigmoid(tf.matmul(a, w2)+biases2)
# y = tf.nn.relu(tf.matmul(a, w2)+biases2)
# y = tf.nn.tanh(tf.matmul(a, w2)+biases2)


#创建一个多隐藏层神经网络
W1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
b1 = tf.Variable(tf.zeros([LAYER1_NODE])+0.1)
L1 = tf.nn.relu(tf.matmul(x, W1)+b1)
#L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
b2 = tf.Variable(tf.zeros([LAYER2_NODE])+0.1)
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
#L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1, seed=1))
b3 = tf.Variable(tf.zeros([OUTPUT_NODE])+0.1)
#y = tf.nn.relu(tf.matmul(L2, W3)+b3)

# 定义代价函数
#cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
cross_entropy = tf.reduce_mean(tf.square(y_ - y))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

# 反向传播算法
train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(LEARNING_RATE_BASE).minimize(cross_entropy)

X = inputMat
Y = outputMat[:, 0:4]


#定义一个用来存储回话的方法，用来存取权值
saver = tf.train.Saver()
with tf.Session() as sess:

    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if isTrain:
        # 训练模型。
        for i in range(TRAINING_STEPS+1):
            for batch in range(n_batch):
                batch_xs = X[batch:batch + BATCH_SIZE, :]
                batch_ys = Y[batch:batch + BATCH_SIZE, :]
                start = (i * BATCH_SIZE) % 5000
                end = (i * BATCH_SIZE) % 5000 + BATCH_SIZE
                # 执行模型训练的方法   传入相应的数据
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 100 == 0:
                # 获取一组训练数据
                check_x = mat(check_input[0, :])
                check_y = mat(check_output[0, 0:4])
                print(check_y)
                yy = sess.run(y, feed_dict={x: check_x})
                print(yy)
                print("***************************************************")
