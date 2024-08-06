import numpy as np


class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, name = 'BN'):
        self.gamma = np.ones((1, num_features, 1, 1))  # 修改维度以匹配输入的维度
        self.beta = np.zeros((1, num_features, 1, 1))
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        self.name = name

    def get_parameters(self):
        return {
            'type': 'BatchNormalization',
            'config': {
                'num_features': self.gamma.shape[1],
                'eps': self.eps,
                'momentum': self.momentum
            },
            'gamma': self.gamma,
            'beta': self.beta,
            'running_mean': self.running_mean,
            'running_var': self.running_var
        }

    def set_parameters(self, params):
        self.gamma = params['gamma']
        self.beta = params['beta']
        self.running_mean = params['running_mean']
        self.running_var = params['running_var']

    def forward(self, X, is_training=True):
        if is_training:
            batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # 计算均值
            batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)  # 计算方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.eps)
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * X_norm + self.beta
        return out

    def backward(self, grads):

        # print("Batch Norm Gamma shape:", self.gamma.shape)
        # print("Input grads shape:", grads.shape)

        X, X_norm, mean, var = self.cache
        N, C, H, W = X.shape  # 获取输入维度

        mean = mean.reshape(1, C, 1, 1)  # 确保维度匹配，这应该已经在forward中处理，可以省略
        var = var.reshape(1, C, 1, 1)  # 同上

        X_mu = X - mean
        std_inv = 1. / np.sqrt(var + self.eps)

        dX_norm = grads * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=(0, 2, 3), keepdims=True) * -.5 * std_inv ** 3
        dmean = np.sum(dX_norm * -std_inv, axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2. * X_mu, axis=(0, 2, 3),
                                                                                           keepdims=True)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / (N * H * W)) + (dmean / (N * H * W))
        self.dgamma = np.sum(grads * X_norm, axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)
        self.dbeta = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, C, 1, 1)

        return dX

