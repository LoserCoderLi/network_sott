import numpy as np
from layers.Conv_Pool import ConvolutionLayer  # 导入卷积层类
from layers.BatchNorm import BatchNormalization  # 导入批量归一化层类
from layers.Linear_Act import Activation  # 导入激活函数类

# 残差块类定义
class ResidualBlock:
    def __init__(self, input_channels, output_channels, opm=None, stride=1, use_batch_norm=False):
        # 初始化第一个卷积层
        self.conv1 = ConvolutionLayer(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, optimizer=opm)
        # 如果使用批量归一化，则初始化
        self.bn1 = BatchNormalization(output_channels) if use_batch_norm else None
        # 初始化ReLU激活函数
        self.relu = Activation('relu')
        # 初始化第二个卷积层
        self.conv2 = ConvolutionLayer(output_channels, output_channels, kernel_size=3, stride=1, padding=1, optimizer=opm)
        # 如果使用批量归一化，则初始化
        self.bn2 = BatchNormalization(output_channels) if use_batch_norm else None
        self.input_channels = input_channels  # 输入通道数
        self.output_channels = output_channels  # 输出通道数
        self.stride = stride  # 步长
        self.use_batch_norm = use_batch_norm  # 是否使用批量归一化

        # 如果步长不为1或输入通道数不等于输出通道数，则初始化捷径连接的卷积层
        if stride != 1 or input_channels != output_channels:
            self.shortcut = ConvolutionLayer(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, optimizer=opm)
        else:
            self.shortcut = None  # 否则捷径连接为空

    # 获取模型参数
    def get_parameters(self):
        params = {
            'type': 'ResidualBlock',  # 模型类型
            'config': {
                'input_channels': self.input_channels,  # 输入通道数
                'output_channels': self.output_channels,  # 输出通道数
                'stride': self.stride,  # 步长
                'use_batch_norm': self.use_batch_norm  # 是否使用批量归一化
            },
            'conv1': self.conv1.get_parameters(),  # 第一个卷积层参数
            'conv2': self.conv2.get_parameters(),  # 第二个卷积层参数
            'relu': self.relu.get_parameters(),  # ReLU激活函数参数
        }
        # 如果使用了批量归一化，则添加相应参数
        if self.bn1:
            params['bn1'] = self.bn1.get_parameters()
        if self.bn2:
            params['bn2'] = self.bn2.get_parameters()
        # 如果有捷径连接卷积层，则添加其参数
        if self.shortcut:
            params['shortcut'] = self.shortcut.get_parameters()
        return params

    # 设置模型参数
    def set_parameters(self, params):
        try:
            # 设置卷积层和激活函数的参数
            self.conv1.set_parameters(params['conv1'])
            self.conv2.set_parameters(params['conv2'])
            # 如果使用了批量归一化，则设置相应参数
            if 'bn1' in params and self.bn1:
                self.bn1.set_parameters(params['bn1'])
            if 'bn2' in params and self.bn2:
                self.bn2.set_parameters(params['bn2'])
            # 如果有捷径连接卷积层，则设置其参数
            if 'shortcut' in params and self.shortcut:
                self.shortcut.set_parameters(params['shortcut'])
        except KeyError as e:
            # 如果缺少参数，则抛出异常
            raise KeyError(f"Missing parameter for ResidualBlock: {e}")

    # 前向传播函数
    def forward(self, X, is_training=True):
        # 第一层卷积
        out = self.conv1.forward(X)
        # 如果使用批量归一化，则进行归一化操作
        if self.bn1:
            out = self.bn1.forward(out, is_training)
        # ReLU激活
        out = self.relu.forward(out)

        # 第二层卷积
        out = self.conv2.forward(out)
        # 如果使用批量归一化，则进行归一化操作
        if self.bn2:
            out = self.bn2.forward(out, is_training)

        # 捷径连接，如果没有额外的卷积层，则直接连接输入
        shortcut = X if self.shortcut is None else self.shortcut.forward(X)
        # 将卷积结果与捷径连接结果相加
        out += shortcut
        # 再次通过ReLU激活
        out = self.relu.forward(out)

        # 缓存输入和输出，用于反向传播
        self.cache = (X, shortcut, out)
        return out

    # 反向传播函数
    def backward(self, grads):
        X, shortcut, out = self.cache

        # 对输出进行反向激活
        grad_out = grads * self.relu.backward(out)

        # 第二层卷积的反向传播
        grad_out2 = self.conv2.backward(grad_out)
        # 如果使用了批量归一化，则进行反向传播
        if self.bn2:
            grad_out2 = self.bn2.backward(grad_out2)

        # 第一层卷积的反向传播
        grad_out1 = self.conv1.backward(grad_out2)
        # 如果使用了批量归一化，则进行反向传播
        if self.bn1:
            grad_out1 = self.bn1.backward(grad_out1)

        # 捷径连接的反向传播，如果有捷径卷积层，则进行其反向传播
        grad_shortcut = grad_out if self.shortcut is None else self.shortcut.backward(grad_out)

        # 返回总的梯度
        return grad_out1 + grad_shortcut

# 以下为一个卷积层类的示例代码（未使用）

# class ConvolutionLayerN:
#     def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, Conv=False):
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
#         self.biases = np.zeros(output_channels)
#         self.Conv = Conv
#         self.input = 0
#
#     def forward(self, input):
#         self.input = input
#         col = im2col(input, self.kernel_size, self.stride, self.padding)
#         col_weights = self.weights.reshape(self.output_channels, -1)
#         out = np.dot(col, col_weights.T) + self.biases
#         out_h = (input.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
#         out_w = (input.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
#         out = out.reshape(input.shape[0], out_h, out_w, self.output_channels).transpose(0, 3, 1, 2)
#         return out
#
#     def backward(self, dout):
#         col = im2col(self.input, self.kernel_size, self.stride, self.padding)
#         dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, self.output_channels)
#         self.weights_gradient = np.dot(col.T, dout_reshaped).reshape(self.weights.shape)
#         self.biases_gradient = np.sum(dout_reshaped, axis=0)
#         dcol = np.dot(dout_reshaped, self.weights.reshape(self.output_channels, -1))
#         dinput = col2im(dcol, self.input.shape, self.kernel_size, self.stride, self.padding)
#         return dinput
#
#     def update(self, learning_rate):
#         self.weights -= learning_rate * self.weights_gradient
#         self.biases -= learning_rate * self.biases_gradient
