
import tensorflow as tf
from numpy.random import RandomState
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import *   # 导入numpy的库函数
import matplotlib
# 解决画图乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
isTrain = True

# 输入 matlab 训练数据文件
train_data = sio.loadmat('D:\\MATLAB Workspace\\gradutionold\\data\\trainData_100000_maxmin.mat')
# 输入 matlab 检验数据文件
check_data = sio.loadmat('D:\\MATLAB Workspace\\gradutionold\\data\\checkdata3000_01_09package.mat')

# 设置相关的参数
INPUT_NODE = 100  # 输入层节点数
OUTPUT_START = 0  # 输出起始点
OUTPUT_END = 15  # 输出结束点
OUTPUT_NODE = OUTPUT_END - OUTPUT_START  # 输出层节点数

LAYER1_NODE = 150  # 第一隐藏层节点数
LAYER2_NODE = 100  # 第二隐藏层节点数
LAYER3_NODE = 80
LAYER4_NODE = 70

TRAINING_STEPS = 50  # 训练轮数

LEARNING_RATE_BASE = 0.001  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
MOVING_AVERAGE_DECAY = 0.98
keep_prob = 1  # 神经元正则化参数，神经元工作百分比

# 分别获取训练数据的输入和输出，作为神经网络的参数
inputMat = train_data['inputMat01']
outputMat = train_data['outputMat01']

check_input = check_data['inputMat']
check_output_all = check_data['outputMat']
check_output = check_output_all[:, OUTPUT_START:OUTPUT_END]

# 计算输入数据的维度
m, n = inputMat.shape
plt.figure(1)
for batchSize in range(100, 100, 10000):
    BATCH_SIZE = batchSize  # 训练数据批次大小700，400（94%），500（94%）
    print(BATCH_SIZE)
    # 计算批次总数
    n_batch = m // BATCH_SIZE
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE), name="x-input")
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    # 生成神经网络的权重和偏执值（单隐藏层）
    w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.zeros([LAYER1_NODE]) + 0.1)
    biases2 = tf.Variable(tf.zeros([OUTPUT_NODE]) + 0.1)

    # 搭建神经网络模型（单隐藏层）
    # a = tf.nn.relu(tf.matmul(x, w1)+biases1)
    # y = tf.nn.relu(tf.matmul(a, w2)+biases2)

    # 创建一个二层神经网络
    W1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    b1 = tf.Variable(tf.zeros([LAYER1_NODE]) + 0.1)
    L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    # 正则化神经网络防止过拟合
    L1_drop = tf.nn.dropout(L1, keep_prob)

    W2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    b2 = tf.Variable(tf.zeros([LAYER2_NODE]) + 0.1)
    L2 = tf.nn.relu(tf.matmul(L1_drop, W2) + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob)

    W3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, LAYER3_NODE], stddev=0.1))
    b3 = tf.Variable(tf.zeros([LAYER3_NODE]) + 0.1)
    L3 = tf.nn.relu(tf.matmul(L2_drop, W3) + b3)
    L3_drop = tf.nn.dropout(L3, keep_prob)

    W4 = tf.Variable(tf.truncated_normal([LAYER3_NODE, LAYER4_NODE], stddev=0.1))
    b4 = tf.Variable(tf.zeros([LAYER4_NODE]) + 0.1)
    L4 = tf.nn.relu(tf.matmul(L3_drop, W4) + b4)
    L4_drop = tf.nn.dropout(L4, keep_prob)

    W5 = tf.Variable(tf.truncated_normal([LAYER4_NODE, OUTPUT_NODE], stddev=0.1))
    b5 = tf.Variable(tf.zeros([OUTPUT_NODE]) + 0.1)
    y = tf.nn.relu(tf.matmul(L4_drop, W5) + b5)

    # 定义代价函数，这之间包含几种不同的代价函数
    cross_entropy = tf.reduce_mean(tf.square(y_ - y))
    # 设置学习率衰减
    # LEARNING_RATE_BASE为初始学习率；
    # 100表示每100轮学习率变一次,这个轮数是由total_data除以batch_size得到的，也可以理解为每一个epoch学习率变一次；
    # LEARNING_RATE_DECAY表示每次变化为上一次学习率乘以LEARNING_RATE_DECAY；
    # staircase= True表示成阶梯函数下降，False时表示连续衰减。
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, n_batch, LEARNING_RATE_DECAY,
                                               staircase=False)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)

    X = inputMat
    Y = outputMat[:, OUTPUT_START:OUTPUT_END]

    # 定义一个用来存储回话的方法，用来存取权值
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if isTrain:
            # 循环搜索最佳batchsize的值

            # 训练模型
            # 定义一个数组，存储一次训练后的平均准确率
            print(batchSize)
            all_accuracy_mean = []
            BATCH_SIZE = batchSize
            for i in range(TRAINING_STEPS + 1):

                # ---------每轮训练结束后重新打乱数据集--------------
                index = [i for i in range(len(X))]
                random.shuffle(index)
                newInput = X[index]
                label = Y[index]
                # -----------打乱数据集完成-------------------------------------

                for batch in range(n_batch):
                    batch_xs = newInput[batch:batch + BATCH_SIZE, :]
                    batch_ys = label[batch:batch + BATCH_SIZE, :]
                    start = (i * BATCH_SIZE) % 5000
                    end = (i * BATCH_SIZE) % 5000 + BATCH_SIZE
                    # 执行模型训练的方法   传入相应的数据
                    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

                if i % 1 == 0:
                    # 计算在验证数据集上的准确率
                    data_check_x = check_input
                    data_check_y = check_output
                    prediction = sess.run(y, feed_dict={x: data_check_x})
                    accuracy_mean = tf.reduce_mean(1 - (tf.abs(prediction - data_check_y) / data_check_y))
                    meanNum = sess.run(accuracy_mean)
                    print(meanNum)
                    all_accuracy_mean.append(meanNum)
            point = tf.reduce_mean(all_accuracy_mean)
            print(batchSize, point)
            print("***********************************************************")
            plt.scatter(batchSize, point, color='red')
plt.title('平均准确率搜索曲线')
plt.show()