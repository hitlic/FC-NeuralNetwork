# coding: utf-8
# author: LiuChen

from neural_network import *
from mnist import MNIST
from data_tools import *

# 加载mnist数据集
mnistdata = MNIST('./mnist_dataset')
mnistdata.gz = True
train_dev_set = mnistdata.load_training()
test_set = mnistdata.load_testing()

# 训练集
train_x = train_dev_set[0][:55000]
train_y = train_dev_set[1][:55000]
train_y = one_hot(train_y)
# 验证集
dev_x = train_dev_set[0][55000:]
dev_y = train_dev_set[1][55000:]
dev_y = one_hot(dev_y)
# 测试集
test_x = test_set[0]
test_y = test_set[1]
test_y = one_hot(test_y)

# 神经网络结构定义
network = FCNetwork(lmd=0.1)  # 学习率
act_fun = Relu  # 隐含层激活函数，Sigmoid、Tanh 或 Relu

# 输入层
input_layer = Layer(784, is_input=True)
network.add_layer(input_layer)

# 隐含层1
hidden1 = Layer(256, activate_fun=act_fun)
network.add_layer(hidden1)

# 隐含层2
hidden2 = Layer(128, activate_fun=act_fun)
network.add_layer(hidden2)

# 隐含层3
hidden3 = Layer(64, activate_fun=act_fun)
network.add_layer(hidden3)

# 输出层
output = Layer(10, activate_fun=act_fun)
network.add_layer(output)

network.init('dims')  # random、he、xavier1、xavier2、dims或normal
network.set_loss(CrossEntropyWithSoftmax)  # 损失函数，均方误差损失为 MSELoss

# 训练
precs, grad, loss = network.train(train_x, train_y, dev_x, dev_y, 128, 50)  # min-batch 128, 迭代次数 50

# 测试
p = network.test(test_x, test_y)
print("测试准确率", p)

# 保存模型
# save_model(network, './mnist_model.pk')

# 加载模型
# network = load_model('./mnist_model.pk')

# 画图显示结果
import matplotlib.pyplot as plt
fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体，否则中文乱码

plt.subplot(3, 1, 1)  # 上图，2行2列第1幅图
plt.title("Test Precision:"+str(p), fontsize=15)
plt.plot(precs, color='g', label="precision")
plt.legend(loc='upper right', frameon=True)

plt.subplot(3, 1, 2)  # 上图，2行2列第1幅图
plt.plot(grad, color='r', label="gradien")
plt.legend(loc='upper right', frameon=True)

plt.subplot(3, 1, 3)  # 下图，2行2列第2幅图
plt.plot(loss, color='b', label="loss")
plt.legend(loc='upper right', frameon=True)

plt.show()
