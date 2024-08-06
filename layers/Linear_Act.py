## Last vision
import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, optimizer, init_method='he', lr_schedule='normal', name='fc'):
        # fan_in = input_dim
        # fan_out = output_dim
        # self.weights = np.random.randn(input_dim, output_dim) / np.sqrt(fan_in)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name  # 添加 name 属性
        std = np.sqrt(2. / input_dim)
        # self.weights = np.random.randn(self.input_dim, self.output_dim) * std
        self.weights = self.initialize_weights(init_method)
        # print(self.weights)
        self.biases = np.zeros(self.output_dim)
        self.optimizer = optimizer

        # lr_scheduler = LearningRateScheduler(initial_lr=0.01, schedule_type=lr_schedule, warmup_steps=100,
        #                                      total_steps=10000)

    def initialize_weights(self, method):
        shape = (self.input_dim, self.output_dim)
        if method == 'random':
            return self.random_initialization(shape)
        elif method == 'zero':
            return self.zero_initialization(shape)
        elif method == 'uniform':
            return self.uniform_initialization(shape, -0.1, 0.1)
        elif method == 'normal':
            return self.normal_initialization(shape, 0, 0.01)
        elif method == 'xavier':
            return self.xavier_initialization(shape)
        elif method == 'he':
            return self.he_initialization(shape)
        elif method == 'scaled_normal':
            return self.scaled_normal_initialization(shape, 0.1)
        else:
            raise ValueError("Unknown initialization method")

    def random_initialization(self, shape):
        # 随机初始化，生成0到1之间的随机数
        return np.random.random(shape)

    def zero_initialization(self, shape):
        # 零初始化，所有权重设置为0
        return np.zeros(shape)

    def uniform_initialization(self, shape, low, high):
        # 均匀分布初始化，在指定范围内生成均匀分布的随机数
        return np.random.uniform(low, high, shape)

    def normal_initialization(self, shape, mean, std):
        # 正态分布初始化，根据给定的均值和标准差生成服从正态分布的随机数
        return np.random.normal(mean, std, shape)

    def xavier_initialization(self, shape):
        # Xavier初始化，根据输入和输出的节点数，采用正态分布生成权重
        in_dim = shape[0]  # input_dim
        std = np.sqrt(2 / (in_dim + shape[1]))  # input_dim + output_dim
        return np.random.normal(0, std, shape)

    def he_initialization(self, shape):
        # He初始化，专为ReLU激活函数设计的初始化方法，采用正态分布生成权重
        in_dim = shape[0]  # input_dim
        std = np.sqrt(2 / in_dim)
        return np.random.normal(0, std, shape)

    def scaled_normal_initialization(self, shape, scale):
        return np.random.randn(*shape) * scale

    def get_parameters(self):
        return {
            'type': 'FullyConnectedLayer',
            'config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim
            },
            'weights': self.weights,
            'biases': self.biases
        }

    def set_weights(self, weights):
        self.weights = weights[0]
        self.biases = weights[1]
        print(f"Weights for layer {self.name} set successfully.")

    def set_parameters(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

    def forward(self, input_data):

        self.input_data = input_data
        # print("Forward - Input shape:", input_data.shape)
        if input_data.ndim > 2:
            input_data = input_data.reshape(input_data.shape[0], -1)
        output_data = np.dot(input_data, self.weights) + self.biases
        # print("Forward - Output shape:", output_data.shape)
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grads):
        # print("Backward - Grads shape:", grads.shape)
        self.grad_w = np.dot(self.input_data.T, grads)
        self.grad_b = np.sum(grads, axis=0)
        backward_grads = np.dot(grads, self.weights.T)
        print('Weights gradient max:', np.max(self.grad_w))
        print('Biases gradient max:', np.max(self.grad_b))
        # 打印传递到前一层的梯度形状
        # print("Backward - Backward Grads shape:", backward_grads.shape)
        return np.dot(grads, self.weights.T)

    def update(self):
        # weights_gradient = clip_gradients(self.grad_w)
        # biases_gradient = clip_gradients(self.grad_b)

        self.weights = self.optimizer.update(self.weights, self.grad_w)
        self.biases = self.optimizer.update(self.biases, self.grad_b)

def clip_gradients(grad, threshold=30.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad

class Activation:
    '''
    在神经网络中，选择适当的激活函数对于模型的性能和训练效率至关重要。不同的激活函数有不同的特性，适用于不同的场景。以下是您定义的激活函数的使用指南：

    ### 1. ReLU (Rectified Linear Unit)
    - **适用场景**: 用于隐藏层。
    - **特点**: ReLU函数在正数区域内保持线性，这使得梯度不会饱和，从而加速梯度下降过程。但在负数区域，ReLU的输出为0，这可能导致死亡神经元问题。
    - **使用时机**: 当您需要一个简单而有效的非线性激活函数时。

    ### 2. Sigmoid
    - **适用场景**: 用于二分类问题的输出层。
    - **特点**: Sigmoid函数输出介于0和1之间，适合表示概率。但它在输入值很大或很小的时候会饱和，导致梯度消失问题。
    - **使用时机**: 当您的任务是二分类，且需要输出概率时。

    ### 3. Tanh (Hyperbolic Tangent)
    - **适用场景**: 用于隐藏层。
    - **特点**: Tanh函数输出介于-1和1之间，相比Sigmoid函数，其输出更加标准化（零中心化）。但它也会在输入值较大或较小时饱和。
    - **使用时机**: 当您需要一个零中心化的激活函数时。

    ### 4. Softmax
    - **适用场景**: 用于多分类问题的输出层。
    - **特点**: Softmax函数将输出转换为概率分布，适合多类分类问题。
    - **使用时机**: 当您的任务是多分类，且需要得到一个概率分布时。

    ### 5. Leaky ReLU
    - **适用场景**: 用于隐藏层。
    - **特点**: Leaky ReLU是ReLU的改进版本，它在负数区域内提供一个小的正斜率，从而解决死亡神经元的问题。
    - **使用时机**: 当您担心ReLU可能导致死亡神经元问题时。

    ### 总结
    - **隐藏层**：通常使用**ReLU**或**Leaky ReLU**，但如果需要零中心化可以考虑**Tanh**。
    - **二分类问题的输出层**：通常使用**Sigmoid**。
    - **多分类问题的输出层**：通常使用**Softmax**。

    选择激活函数时，您还需要考虑您的网络结构和数据的特性。实际上，经验和实验常常是选择激活函数的最佳指导。
    '''
    def __init__(self, activation, name = 'act'):
        self.activation = activation
        self.input_data = None
        self.leaky_slope = 0.1
        self.name = name

    def get_parameters(self):
        return {
            'type': 'Activation',
            'config': {
                'activation_type': self.activation
            }
        }
    def set_parameters(self, params):
        # 这里不需要做任何事，因为激活层没有可学习的参数
        pass

    def set_weights(self, weights):
        # Activation layers typically do not have weights to set
        pass

    def forward(self, data):
        self.input_data = data

        if self.activation == 'relu':
            return np.maximum(0, data)

        elif self.activation == 'sigmoid':
            self.input_data = np.clip(self.input_data, -100, 100)
            return np.where(self.input_data >= 0,
                            1 / (1 + np.exp(-self.input_data)),
                            np.exp(self.input_data) / (1 + np.exp(self.input_data)))

        elif self.activation == 'tanh':
            return np.tanh(data)

        elif self.activation == 'leaky_relu':
            return np.where(data >= 0, data, data * self.leaky_slope)

        elif self.activation == 'softmax':
            # 添加调试信息
            print(f"Data shape before softmax: {data.shape}")
            exp_data = np.exp(data - np.max(data, axis=1, keepdims=True))
            return exp_data / np.sum(exp_data, axis=1, keepdims=True)

        else:

            raise ValueError("Unsupported activation function")



    def backward(self, grads):
        if self.activation == 'relu':
            return grads * (self.input_data > 0)

        elif self.activation == 'sigmoid':
            sigmoid_output = self.forward(self.input_data)
            return grads * sigmoid_output * (1 - sigmoid_output)

        elif self.activation == 'tanh':
            return grads * (1 - np.square(np.tanh(self.input_data)))


        elif self.activation == 'softmax':

            softmax_output = self.forward(self.input_data)

            return softmax_output * (1 - softmax_output) * grads

        elif self.activation == 'leaky_relu':
            leaky_grads = np.where(self.input_data >= 0, 1, self.leaky_slope)
            return grads * leaky_grads