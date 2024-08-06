import numpy as np
from layers.Conv_Pool import ConvolutionLayer
from layers.BatchNorm import BatchNormalization
from layers.Linear_Act import Activation


class ResidualBlock:
    def __init__(self, input_channels, output_channels, opm=None, stride=1, use_batch_norm=False):
        self.conv1 = ConvolutionLayer(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, optimizer=opm)
        self.bn1 = BatchNormalization(output_channels) if use_batch_norm else None
        self.relu = Activation('relu')
        self.conv2 = ConvolutionLayer(output_channels, output_channels, kernel_size=3, stride=1, padding=1, optimizer=opm)
        self.bn2 = BatchNormalization(output_channels) if use_batch_norm else None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_batch_norm = use_batch_norm

        if stride != 1 or input_channels != output_channels:
            self.shortcut = ConvolutionLayer(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, optimizer=opm)
        else:
            self.shortcut = None

    def get_parameters(self):
        params = {
            'type': 'ResidualBlock',
            'config': {
                'input_channels': self.input_channels,
                'output_channels': self.output_channels,
                'stride': self.stride,
                'use_batch_norm': self.use_batch_norm
            },
            'conv1': self.conv1.get_parameters(),
            'conv2': self.conv2.get_parameters(),
            'relu': self.relu.get_parameters(),
        }
        if self.bn1:
            params['bn1'] = self.bn1.get_parameters()
        if self.bn2:
            params['bn2'] = self.bn2.get_parameters()
        if self.shortcut:
            params['shortcut'] = self.shortcut.get_parameters()
        return params

    def set_parameters(self, params):
        try:
            self.conv1.set_parameters(params['conv1'])
            self.conv2.set_parameters(params['conv2'])
            if 'bn1' in params and self.bn1:
                self.bn1.set_parameters(params['bn1'])
            if 'bn2' in params and self.bn2:
                self.bn2.set_parameters(params['bn2'])
            if 'shortcut' in params and self.shortcut:
                self.shortcut.set_parameters(params['shortcut'])
        except KeyError as e:
            raise KeyError(f"Missing parameter for ResidualBlock: {e}")

    def forward(self, X, is_training=True):
        out = self.conv1.forward(X)
        if self.bn1:
            out = self.bn1.forward(out, is_training)
        out = self.relu.forward(out)

        out = self.conv2.forward(out)
        if self.bn2:
            out = self.bn2.forward(out, is_training)

        shortcut = X if self.shortcut is None else self.shortcut.forward(X)
        out += shortcut
        out = self.relu.forward(out)

        self.cache = (X, shortcut, out)
        return out

    def backward(self, grads):
        X, shortcut, out = self.cache

        grad_out = grads * self.relu.backward(out)
        # print("grad_out:", grad_out.shape)

        grad_out2 = self.conv2.backward(grad_out)
        # print("grad_out2:", grad_out2.shape)
        if self.bn2:
            grad_out2 = self.bn2.backward(grad_out2)
            # print("grad_out2(bn2):", grad_out2.shape)

        grad_out1 = self.conv1.backward(grad_out2)
        # print("grad_out1:", grad_out1.shape)
        if self.bn1:
            grad_out1 = self.bn1.backward(grad_out1)
            # print("grad_out1(bn1):", grad_out1.shape)

        grad_shortcut = grad_out if self.shortcut is None else self.shortcut.backward(grad_out)

        return grad_out1 + grad_shortcut

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
