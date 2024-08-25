from numpy.lib.stride_tricks import as_strided
import numpy as np
from training.Optimizer import LearningRateScheduler


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
        """
    将输入的多维图像数据转换为二维矩阵，用于卷积运算的高效实现。
    
    参数:
    - input_data: 输入的图像数据，形状为 (N, C, H, W)
    - filter_h: 滤波器的高度
    - filter_w: 滤波器的宽度
    - stride: 滤波器应用的步长，默认为 1
    - pad: 图像边缘的填充量，默认为 0
    
    返回:
    - col: 转换后的二维矩阵，形状为 (N * out_h * out_w, C * filter_h * filter_w)
    """
    N, C, H, W = input_data.shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    if pad > 0:
        input_data = np.pad(input_data, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    shape = (N, C, out_h, out_w, filter_h, filter_w)
    strides = input_data.strides[:2] + (input_data.strides[2] * stride, input_data.strides[3] * stride) + input_data.strides[2:]
    col = as_strided(input_data, shape=shape, strides=strides)
    col = col.reshape(N * out_h * out_w, C * filter_h * filter_w)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """
    将二维矩阵转换回多维图像数据，这是 im2col 操作的逆过程。
    
    参数:
    - col: 经过 im2col 转换的二维矩阵
    - input_shape: 输入的图像数据的形状 (N, C, H, W)
    - filter_h: 滤波器的高度
    - filter_w: 滤波器的宽度
    - stride: 滤波器应用的步长，默认为 1
    - pad: 图像边缘的填充量，默认为 0
    
    返回:
    - img: 还原后的多维图像数据
    """
    N, C, H, W = input_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    out_h = (H_padded - filter_h) // stride + 1
    out_w = (W_padded - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)

    for i in range(filter_h):
        i_max = i + stride * out_h
        for j in range(filter_w):
            j_max = j + stride * out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]

    if pad > 0:
        return img[:, :, pad:-pad, pad:-pad]
    else:
        return img


class ConvolutionLayer:
    def __init__(self, input_channels, output_channels, kernel_size, optimizer, stride=1, padding=0, Conv = False, init_method='he', lr_schedule='normal', name='conv'):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.weights = self.initialize_weights(init_method)
        self.biases = np.zeros(output_channels)  # 为了广播的方便，保持一维
        self.col_cache = None  # 缓存im2col结果以重用
        self.Conv = Conv
        self.name = name  # 添加 name 属性
        # print("原来的we",self.weights)
        #
        #
        # print("原来的bia", self.biases)

        lr_scheduler = LearningRateScheduler(initial_lr=0.01, schedule_type=lr_schedule, warmup_steps=100,
                                             total_steps=10000)
        self.optimizer = optimizer


    def initialize_weights(self, method):
        shape = (self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
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
        return np.random.random(shape)

    def zero_initialization(self, shape):
        return np.zeros(shape)

    def uniform_initialization(self, shape, low, high):
        return np.random.uniform(low, high, shape)

    def normal_initialization(self, shape, mean, std):
        return np.random.normal(mean, std, shape)

    def xavier_initialization(self, shape):
        in_dim = shape[1] * shape[2] * shape[3]  # input_channels * kernel_size * kernel_size
        std = np.sqrt(2 / (in_dim + shape[0]))  # input_channels + output_channels
        return np.random.normal(0, std, shape)

    def he_initialization(self, shape):
        in_dim = shape[1] * shape[2] * shape[3]  # input_channels * kernel_size * kernel_size
        std = np.sqrt(2 / in_dim)
        return np.random.normal(0, std, shape)

    def scaled_normal_initialization(self, shape, scale):
        return np.random.randn(*shape) * scale

    def get_parameters(self):
        return {
            'type': 'ConvolutionLayer',
            'config': {
                'input_channels': self.input_channels,
                'output_channels': self.output_channels,
                'kernel_size': self.kernel_size,
                'stride': self.stride,
                'padding': self.padding
            },
            'weights': self.weights,
            'biases': self.biases
        }

    def set_weights(self, weights):
        self.weights = weights[0]
        self.biases = weights[1]
        # print(self.biases)
        # print(self.weights)

        print(f"Weights for layer {self.name} set successfully.")


    def set_parameters(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

    def forward(self, input):
        self.input = input
        # print(input.shape)

        self.col_cache = im2col(input, self.kernel_size, self.kernel_size, self.stride, self.padding)
        self.col_weights = self.weights.reshape(self.output_channels, -1)
        out = np.dot(self.col_cache, self.col_weights.T) + self.biases
        out_h = (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out.reshape(input.shape[0], out_h, out_w, self.output_channels).transpose(0, 3, 1, 2)
        # print("o", out.shape)
        return out

    def backward(self, dout):
        dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.output_channels)
        col = self.col_cache
        self.weights_gradient = np.dot(self.col_cache.T, dout_reshaped).transpose(1, 0).reshape(self.output_channels,self.input_channels, self.kernel_size,self.kernel_size)
        self.biases_gradient = np.sum(dout_reshaped, axis=0)

        dcol = np.dot(dout_reshaped, self.col_weights)
        dinput = col2im(dcol, self.input.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)

        # print('Weights gradient max:', np.max(self.weights_gradient))
        # print('Biases gradient max:', np.max(self.biases_gradient))

        return dinput

    def update(self):
        self.weights_gradient = clip_gradients(self.weights_gradient)
        self.biases_gradient = clip_gradients(self.biases_gradient)
        self.weights = self.optimizer.update(self.weights, self.weights_gradient)
        self.biases = self.optimizer.update(self.biases, self.biases_gradient)
        # print('Weights max after update:', np.max(self.weights))
        # print('Biases max after update:', np.max(self.biases))

def clip_gradients(grad, threshold=30.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad


class MaxPoolingLayer:
    def __init__(self, pool_size, stride, name = 'pool'):
        self.pool_size = pool_size
        self.stride = stride
        self.name = name

    def set_weights(self, weights):
        # MaxPoolingLayer typically does not have weights to set
        pass

    def get_parameters(self):
        return {
            'type': 'MaxPoolingLayer',
            'config': {
                'pool_size': self.pool_size,
                'stride': self.stride
            }
        }
    def set_parameters(self, params):
        # 虽然池化层没有可学习的参数，但可以检查并设置配置，如果有需要的话
        pass

    def forward(self, input):
        self.input = input
        # print(input.shape)
        batch_size, channels, height, width = input.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                input_slice = input[:, :, h_start:h_end, w_start:w_end]
                max_pool_indices = np.argmax(input_slice.reshape(batch_size, channels, -1), axis=2)
                self.output[:, :, i, j] = np.max(input_slice, axis=(2, 3))

                # Store the indices of max values
                max_indices_row = max_pool_indices // self.pool_size
                max_indices_col = max_pool_indices % self.pool_size
                self.max_indices[:, :, i, j, 0] = h_start + max_indices_row
                self.max_indices[:, :, i, j, 1] = w_start + max_indices_col
        # print("o", self.output.shape)

        return self.output

    def backward(self, output_gradient):
        input_gradient = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = output_gradient.shape

        for i in range(out_height):
            for j in range(out_width):
                max_h_indices = self.max_indices[:, :, i, j, 0]
                max_w_indices = self.max_indices[:, :, i, j, 1]

                for b in range(batch_size):
                    for c in range(channels):
                        h_index = max_h_indices[b, c]
                        w_index = max_w_indices[b, c]
                        input_gradient[b, c, h_index, w_index] += output_gradient[b, c, i, j]

        return input_gradient


class FlattenLayer:
    def __init__(self, name = 'flatten'):
        self.name = name
        pass

    def set_weights(self, weights):
        # FlattenLayer typically does not have weights to set
        pass

    def forward(self, input):
        self.input_shape = input.shape  # 保存输入形状以便于在反向传播时重塑
        # print(input.shape)
        # print((input.reshape(input.shape[0], -1)).shape)
        return input.reshape(input.shape[0], -1)  # 展平除批次维度以外的所有维度

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)  # 将梯度重塑回输入的形状

    def update(self):
        pass  # 展平层不需要更新操作

    def get_parameters(self):
        return {
            'type': 'FlattenLayer'
        }

    def set_parameters(self, params):
        # Flatten layer does not have parameters to set
        pass
