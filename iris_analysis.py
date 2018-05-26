# coding: utf-8
# author: LiuChen

from data_tools import *
from neural_network import *

# 神经网络结构定义
network = FCNetwork(0.1)  # 学习率
act_fun = Tanh  # 隐含层激活函数，Sigmoid、Tanh 或 Relu

# 输入层
input_layer = Layer(4, is_input=True)
network.add_layer(input_layer)

# 隐含层1
hidden1 = Layer(20, activate_fun=act_fun)
network.add_layer(hidden1)

# 隐含层2
hidden2 = Layer(10, activate_fun=act_fun)
network.add_layer(hidden2)

# 输出层
output = Layer(3, activate_fun=act_fun)
network.add_layer(output)

network.init('dims')  # random、he、xavier1、xavier2、dims或normal
network.set_loss(CrossEntropyWithSoftmax)  # 损失函数，均方误差损失为 MSELoss

# 加载数据
dataset = load_iris_data('iris.data.txt')
x = [data[0] for data in dataset]
y = [data[1] for data in dataset]

x_train = x[0: 120]
y_train = y[0: 120]
x_test = x[120:]
y_test = y[120:]

# 训练
precs, grad, loss = network.train(x_train, y_train, x_test, y_test, 30, 500)
# 测试
p = network.test(x_test, y_test)
print("------\n测试准确率", p)

# 保存模型
# save_model(network, './iris_model.pk')

# 加载模型
# network = load_model('./iris_model.pk')

# 画图显示结果
import matplotlib.pyplot as plt
fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体，否则中文乱码

plt.subplot(3, 1, 1)  # 上图，2行2列第1幅图
plt.title("Test Precision:"+str(p), fontsize=15)
plt.plot(precs, color='g', label="precision")
plt.legend(loc='upper right', frameon=True)

plt.subplot(3, 1, 2)  # 上图，2行2列第1幅图
plt.plot(grad, color='r', label="gradient")
plt.legend(loc='upper right', frameon=True)

plt.subplot(3, 1, 3)  # 下图，2行2列第2幅图
plt.plot(loss, color='b', label="loss")
plt.legend(loc='upper right', frameon=True)

plt.show()
