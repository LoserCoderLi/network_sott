import numpy as np

# BatchNormalization类，用于实现批量归一化层
class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, name='BN'):
        # 初始化缩放因子（gamma）和偏移量（beta）
        self.gamma = np.ones((1, num_features, 1, 1))  # 缩放因子，初始化为1，维度与输入匹配
        self.beta = np.zeros((1, num_features, 1, 1))  # 偏移量，初始化为0，维度与输入匹配
        self.eps = eps  # 防止除以0的小常数
        self.momentum = momentum  # 用于计算移动平均的动量
        self.running_mean = np.zeros((1, num_features, 1, 1))  # 初始化运行中的均值
        self.running_var = np.ones((1, num_features, 1, 1))  # 初始化运行中的方差
        self.name = name  # 层的名称

    # 获取层的参数
    def get_parameters(self):
        return {
            'type': 'BatchNormalization',  # 层的类型
            'config': {
                'num_features': self.gamma.shape[1],  # 特征数量
                'eps': self.eps,  # 防止除以0的值
                'momentum': self.momentum  # 动量
            },
            'gamma': self.gamma,  # 缩放因子
            'beta': self.beta,  # 偏移量
            'running_mean': self.running_mean,  # 运行中的均值
            'running_var': self.running_var  # 运行中的方差
        }

    # 设置层的参数
    def set_parameters(self, params):
        self.gamma = params['gamma']  # 设置缩放因子
        self.beta = params['beta']  # 设置偏移量
        self.running_mean = params['running_mean']  # 设置运行中的均值
        self.running_var = params['running_var']  # 设置运行中的方差

    # 前向传播函数
    def forward(self, X, is_training=True):
        if is_training:
            # 计算批次的均值和方差
            batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # 计算均值
            batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)  # 计算方差
            # 更新运行中的均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            # 归一化输入
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.eps)
            self.cache = (X, X_norm, batch_mean, batch_var)  # 缓存数据用于反向传播
        else:
            # 在测试模式下，使用运行中的均值和方差进行归一化
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # 使用gamma和beta进行缩放和平移
        out = self.gamma * X_norm + self.beta
        return out

    # 反向传播函数
    def backward(self, grads):
        # 从缓存中提取前向传播时的数据
        X, X_norm, mean, var = self.cache
        N, C, H, W = X.shape  # 获取输入的维度

        # 确保均值和方差的维度匹配
        mean = mean.reshape(1, C, 1, 1)
        var = var.reshape(1, C, 1, 1)

        X_mu = X - mean  # 计算输入与均值的差
        std_inv = 1. / np.sqrt(var + self.eps)  # 计算标准差的倒数

        # 计算归一化后的输入的梯度
        dX_norm = grads * self.gamma
        # 计算方差的梯度
        dvar = np.sum(dX_norm * X_mu, axis=(0, 2, 3), keepdims=True) * -.5 * std_inv ** 3
        # 计算均值的梯度
        dmean = np.sum(dX_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2. * X_mu, axis=(0, 2, 3),
                                                                                           keepdims=True)

        # 计算输入X的梯度
        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / (N * H * W)) + (dmean / (N * H * W))
        # 计算gamma和beta的梯度
        self.dgamma = np.sum(grads * X_norm, axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)
        self.dbeta = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)

        return dX  # 返回输入X的梯度
