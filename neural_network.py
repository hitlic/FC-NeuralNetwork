# coding: utf-8
# author: LiuChen

import numpy as np
import math
import pickle


class Sigmoid(object):
    """
    Sigmoid激活函数
    """
    @staticmethod
    def fun(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def diff(x):
        return Sigmoid.fun(x)*(1 - Sigmoid.fun(x))


class Tanh(object):
    """
    tanh双曲正切激活函数
    """
    @staticmethod
    def fun(x):
        return np.tanh(x)

    @staticmethod
    def diff(x):
        return 1 - Tanh.fun(x) * Tanh.fun(x)


class Relu(object):
    """
    Relu激活函数
    leaky Relu
    """
    @staticmethod
    def fun(x):
        return (x > 0) * x + (x <= 0) * (0.01 * x)

    @staticmethod
    def diff(x):
        grad = 1. * (x > 0)
        grad[grad == 0] = 0.01
        return grad


class CrossEntropyWithSoftmax(object):
    """
    带softmax的交叉熵损失函数
    """
    @staticmethod
    def fun(y_hat, y):
        yr_hot = CrossEntropyWithSoftmax.softmax(y_hat) * y
        return np.average(- np.log(np.sum(yr_hot, 1)))

    @staticmethod
    def diff(y_hat, y):
        return y_hat - y

    @staticmethod
    def softmax(y_hat):
        e_x = np.exp(y_hat - np.max(y_hat, 0))
        return e_x / e_x.sum(0)


class MSELoss(object):
    """
    均方误差损失函数
    """
    @staticmethod
    def fun(y_hat, y):
        l = sum(np.average(0.5*(y_hat - y)*(y_hat - y), 0))
        return l

    @staticmethod
    def diff(y_hat, y):
        return y_hat - y


class Layer(object):
    """
    神经网络的一层
    """

    def __init__(self, node_num, activate_fun=None, is_input=False):
        if activate_fun is None and is_input is False:
            raise Exception("非输入层必须指定激活函数")
        self.dim = node_num
        self.W = None  # 权值矩阵
        self.dW = None  # 权值矩阵梯度
        self.b = None  # 残差向量
        self.db = None  # 残差向量梯度
        self.z = None  # 当前层的总输入 z=Wa_p +b
        self.a = None  # 当前层前向计算的输出向量 a=activate(z)
        self.delta = None  # 反向传播的delta，即 dC/dz
        self.activate = activate_fun
        self.is_input = is_input  # 当前层是否是输入层
        self.network = None  # 当前层所属的神经网络
        self.prev_layer = None  # 当前层的前一层
        self.next_layer = None  # 当前层的后一层
        self.name = "Layer"

    def set_input(self, x):
        """
        输入层的输出
        """
        if self.is_input:  # 只有输入层能输入
            self.x = x

    def forward(self):
        """
        当前层前向传播
        """
        if self.is_input:
            self.a = self.x  # 输入层的输出等于输入，shape=(dim, data_num)
            return
        self.z = np.dot(self.W, self.prev_layer.a) + self.b  # z = Wa^[l-1] + b; shape=(dim, data_num)
        self.a = self.activate.fun(self.z)  # a = sigma(z); shape=(dim, data_num)

    def backword(self):
        """
        当前层反向传播
        """
        if self.is_input:  # 若为输入层，则不用做任何操作
            return

        if self is self.network[-1]:  # 若为输出层
            self.delta = self.activate.diff(self.z) * self.network.diff_y  # delta=sigma'(z) * dy; shape=(dim,data_num)
        else:
            W_next = self.next_layer.W  # 下一层权值
            trans_expand_next_delta = np.expand_dims(np.transpose(self.next_layer.delta), 2)  # 改变形状以适于批量矩阵运算
            W_next_delta_next = np.matmul(np.transpose(W_next), trans_expand_next_delta)  # mul(W^[l-1], delta^[l-1])
            # a * mul(W^[l-1], delta^[l-1])
            self.delta = self.activate.diff(self.z) * np.transpose(np.squeeze(W_next_delta_next, 2))

        # 求参数梯度
        delta_expand = np.expand_dims(np.transpose(self.delta), 2)  # 改变形状以适于批量矩阵运算
        prev_a_expand = np.expand_dims(np.transpose(self.prev_layer.a), 1)  # 改变形状以适于批量矩阵运算
        self.dW = np.average(np.matmul(delta_expand, prev_a_expand), 0)  # dW=mul(delta,a^[l-1]); shape=(dim,dim^[l-1])
        self.db = np.expand_dims(np.average(self.delta, 1), 1)  # db=delta ; shape=(dim,1)
        self.clip_gradient()  # clipse gradient，防止梯度爆炸

    def clip_gradient(self):
        """
        clip梯度，避免梯度爆炸
        """
        threshold = 1/self.network.lmd
        norm_dW = np.linalg.norm(self.dW)
        norm_db = np.linalg.norm(self.db)
        if norm_dW > threshold:
            self.dW = threshold/norm_dW * self.dW
            print("... ... 权值矩阵梯度 cliped!")
        if norm_db > threshold:
            self.db = threshold/norm_db * self.db
            print("... ... 残差向量梯度 cliped!")

    def greaient_descent(self, lmd):
        if self.is_input:  # 输入层无参数需更新
            return
        # 梯度下降更新参数
        self.W = self.W - lmd * self.dW
        self.b = self.b - lmd * self.db

    def init_prams(self, method):
        """
        随机初始化权值矩阵和残差向量，确定当前层的前一层和后一层
        :param method: random、he、xavier1、xavier2、dims或normal
        """
        self.prev_layer = self.network.prev_layer(self)  # 前一层
        self.next_layer = self.network.next_layer(self)  # 后一层

        if self.is_input:  # 输入层无权值矩阵和残差向量
            return

        if self.W is not None and self.b is not None:  # 如果W和b已存在，则不用再随机初始化
            return

        self.b = np.zeros(shape=[self.dim, 1])  # 初始化残差向量为0向量

        # 多种权值初始化方法
        if method == "random":
            self.W = np.random.randn(self.dim, self.prev_layer.dim)*0.01
        elif method == "he":
            self.W = np.random.randn(self.dim, self.prev_layer.dim)*np.sqrt(2/self.prev_layer.dim)*.01
        elif method == "xavier1":
            self.W = np.random.randn(self.dim, self.prev_layer.dim)*np.sqrt(1/self.prev_layer.dim)*.01
        elif method == "xavier2":
            bound = np.sqrt(6/(self.dim + self.prev_layer.dim))  # 6/sqrt(dim + pre_dim)
            self.W = np.random.uniform(-bound, bound, size=[self.dim, self.prev_layer.dim])
        elif method == "dims":
            bound = np.sqrt(6/(self.dim + self.prev_layer.dim))  # 6/sqrt(dim + pre_dim)
            self.W = np.random.uniform(-bound, bound, size=[self.dim, self.prev_layer.dim])
        elif method == "normal":
            self.W = np.random.normal(size=[self.dim, self.prev_layer.dim])  # 标准正态分布初始化

    def set_params(self, W, b):
        """
        手动设置权值矩阵和残差向量
        """
        if self.is_input:
            raise("输入层无权值矩阵和残差向量")
        self.W = np.array(W)
        self.b = np.array(b)


class FCNetwork(list):
    """
    神经网络，继承自list
    """

    def __init__(self, lmd=2, loss=None):
        self.loss = loss  # 损失函数
        self.diff_y = None  # 输出的梯度
        self.lmd = lmd  # 学习率

    def set_loss(self, loss):
        """
        设置网络的损失函数，运行反向传播前必须设置
        """
        self.loss = loss

    def add_layer(self, layer):
        """
        添加一层（各层按添加先后顺序组合）
        """
        layer.network = self
        layer.name += '-' + str(len(self))
        self.append(layer)

    def init(self, method):
        """
        初始化各层参数
        :param method: random、he、xavier1、xavier2、dims或normal
        """
        for layer in self:
            layer.init_prams(method)

    def forward(self, x):
        """
        前向传播所有层
        """
        if self[-1].W is None:
            raise Exception("请先运神经网络的init方法初始化各层参数")
        x = np.transpose(np.asarray(x))
        self[0].set_input(x)  # 设置输入层的输入 shape=(input_dim, data_num)
        for layer in self:
            # print(".", "*"*30, ".. ...前向"+layer.name)
            layer.forward()  # 逐层前向计算
        return np.transpose(self[-1].a)  # 最后一层的输出结果作为网络的输出 shape=(data_num, output_dim)

    def backword(self, y):
        """
        反向传播所有层
        """
        if self[-1].a is None:
            raise Exception("先运行前向传播forward")
        if self.loss is None:
            raise Exception("没有损失函数")
        y = np.transpose(np.array(y))
        y_hat = self[-1].a
        self.diff_y = self.loss.diff(y_hat, y)  # 输出的梯度
        for layer in reversed(self):
            # print(".", "*"*30, ".. ...前向"+layer.name)
            layer.backword()
        for layer in self:
            layer.greaient_descent(self.lmd)

    def next_layer(self, layer):
        """
        :param layer: 当前层
        :return: 返回当前层的下一层
        """
        if layer is self[-1]:  # 输出层无下一层
            return None
        index = self.index(layer)
        return self[index + 1]

    def prev_layer(self, layer):
        """
        :param layer: 当前层
        :return: 返回当前层的上一层
        """
        if layer is self[0]:  # 输入层无上一层
            return None
        index = self.index(layer)
        return self[index - 1]

    def get_gradient(self):
        """
        网络各层权值矩阵梯度和残差向量梯度的范数和
        """
        grad_sum = 0
        for layer in self:
            if not layer.is_input:
                grad_sum += np.linalg.norm(layer.dW) + np.linalg.norm(layer.db)
        return grad_sum

    def get_loss(self, x, y):
        """
        损失
        """
        out = self.forward(x)
        return self.loss.fun(out, np.array(y))

    def batch_generate(self, data_set, label_set, batch_size):
        """
        把数据集转成minibatch
        """
        size = len(data_set)
        data_set = np.array(data_set)
        label_set = np.array(label_set)
        num_batch = 0
        if size % batch_size == 0:
            num_batch = int(size/batch_size)
        else:
            num_batch = math.ceil(size/batch_size)
        rand_index = list(range(size))
        np.random.shuffle(list(range(size)))
        for i in range(num_batch):
            start = i*batch_size
            end = min((i+1)*(batch_size), size)
            yield data_set[rand_index[start:end]], label_set[rand_index[start:end]]

    def train(self, data_set, label_set, dev_data, dev_label, batch_size=50, epoch=10):
        """
        训练
        """
        grads = []
        losses = []
        precs = []
        for i in range(epoch):
            j = 0
            for data_batch, label_batch in self.batch_generate(data_set, label_set, batch_size):
                j += 1
                print("... 第%d次迭代，第%d个batch" % (i, j))
                self.forward(data_batch)
                self.backword(label_batch)
            precision, grad, loss = self.validate(dev_data, dev_label)
            grads.append(grad)
            losses.append(loss)
            precs.append(precision)
            print("第 %d 次迭代，准确率 %f ，梯度 %f ，损失 %f" % (i, precision, grad, loss))
        return precs, grads, losses

    def validate(self, dev_data, dev_label):
        """
        验证
        """
        grad = self.get_gradient()
        loss = self.get_loss(dev_data, dev_label)
        precision = self.test(dev_data, dev_label)
        return precision, grad, loss

    def test(self, test_data, test_label, batch_size=512):
        """
        测试
        """
        wrong_num = 0
        # 分批测试避免测试数据量太大造成问题
        for data_batch, label_batch in self.batch_generate(test_data, test_label, batch_size):
            predict = self.forward(data_batch)
            wrong_num += np.count_nonzero(np.argmax(predict, 1)-np.argmax(label_batch, 1))
        p = 1 - wrong_num/len(test_data)
        return p

def save_model(model, path):
    """
    保存模型
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    """
    加载模型
    """
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
