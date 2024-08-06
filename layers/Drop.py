import numpy as np

class Dropout:
    def __init__(self, p=0.5, name = 'droput'):
        self.p = p  # Dropout 概率
        self.name = name

    def get_parameters(self):
        return {
            'type': 'Dropout',
            'config': {
                'p': self.p
            }
        }

    def set_parameters(self, params):
        if 'p' in params['config']:  # 确保 'p' 存在于配置字典中
            self.p = params['config']['p']
    def forward(self, X, is_training=True):
        if is_training:
            self.mask = np.random.binomial(1, 1 - self.p, X.shape) / (1 - self.p)
            return X * self.mask
        else:
            return X

    def backward(self, grads):
        return grads * self.mask
