import numpy as np

# Dropout类，用于实现Dropout层
class Dropout:
    def __init__(self, p=0.5, name='dropout'):
        self.p = p  # Dropout的概率，即每个神经元被"丢弃"的概率
        self.name = name  # 层的名称

    # 获取层的参数
    def get_parameters(self):
        return {
            'type': 'Dropout',  # 层的类型
            'config': {
                'p': self.p  # Dropout的概率
            }
        }

    # 设置层的参数
    def set_parameters(self, params):
        # 确保配置字典中包含' p '参数，然后进行设置
        if 'p' in params['config']:
            self.p = params['config']['p']

    # 前向传播函数
    def forward(self, X, is_training=True):
        if is_training:
            # 在训练阶段，随机生成一个与输入形状相同的二项分布掩码
            # 保留的神经元以1/(1-p)的比例缩放，以保持激活值的期望不变
            self.mask = np.random.binomial(1, 1 - self.p, X.shape) / (1 - self.p)
            return X * self.mask  # 将掩码应用于输入X
        else:
            # 在测试阶段，直接返回原始输入，不进行Dropout
            return X

    # 反向传播函数
    def backward(self, grads):
        # 在反向传播阶段，掩码仅应用于梯度
        return grads * self.mask  # 将掩码应用于梯度
