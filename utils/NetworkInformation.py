from layers.Linear_Act import FullyConnectedLayer, Activation  # 导入全连接层和激活函数类
import tkinter as tk  # 导入用于创建GUI应用程序的库
from tkinter import ttk, Label  # 从tkinter中导入ttk模块和Label控件
from tkinter.ttk import Style  # 从ttk模块中导入Style类
from PIL import Image, ImageTk  # 导入PIL库用于图像处理
import os  # 导入操作系统接口模块
import re  # 导入正则表达式模块
from layers.Conv_Pool import ConvolutionLayer, MaxPoolingLayer, FlattenLayer  # 导入卷积层、最大池化层和平坦化层类
from layers.BatchNorm import BatchNormalization  # 导入批量归一化层类
from layers.Residual import ResidualBlock  # 导入残差块类

def print_network_architecture(neural_network):
    """
    打印神经网络的层次结构信息。

    参数:
        neural_network: 包含多个层的神经网络对象。
    """
    print("网络结构:")
    for i, layer in enumerate(neural_network.layers):
        layer_type = type(layer).__name__  # 获取层的类型名称
        print(f"Layer {i + 1}: {layer_type}")
        if isinstance(layer, FullyConnectedLayer):
            # 打印全连接层的信息
            print(f"   Input Dimension: {layer.weights.shape[1]}")
            print(f"   Output Dimension: {layer.weights.shape[0]}")
        elif isinstance(layer, ConvolutionLayer):
            # 打印卷积层的信息
            print(f"   Input Channels: {layer.input_channels}")
            print(f"   Output Channels: {layer.output_channels}")
            print(f"   Kernel Size: {layer.kernel_size}x{layer.kernel_size}")
            print(f"   Stride: {layer.stride}")
            print(f"   Padding: {layer.padding}")
        elif isinstance(layer, MaxPoolingLayer):
            # 打印最大池化层的信息
            print(f"   Pool Size: {layer.pool_size}x{layer.pool_size}")
            print(f"   Stride: {layer.stride}")
        elif isinstance(layer, Activation):
            # 打印激活函数的信息
            print(f"   Activation Function: {layer.activation}")
        elif isinstance(layer, BatchNormalization):
            # 打印批量归一化层的信息
            print(f"   Number of Features: {layer.gamma.shape[1]}")
            print(f"   Epsilon: {layer.eps}")
            print(f"   Momentum: {layer.momentum}")
        elif isinstance(layer, FlattenLayer):
            # 打印平坦化层的信息
            print("   Flattens the input.")
        elif isinstance(layer, ResidualBlock):
            # 打印残差块的信息
            print(f"   Input Channels: {layer.conv1.input_channels}")
            print(f"   Output Channels: {layer.conv2.output_channels}")
            print(f"   Stride: {layer.conv1.stride}")
            # print(f"   Use Batch Norm: {layer.bn1 is not None}")
        print()

    # 打印优化器和损失函数信息
    if hasattr(neural_network, 'optimizer_name') and hasattr(neural_network, 'loss_name'):
        print(f"Optimizer: {neural_network.optimizer_name}")
        print(f"Loss Function: {neural_network.loss_name}")

def natural_sort_key(s):
    """提取文本中的数字用于自然排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class MetricsViewer:
    def __init__(self, master, image_folder):
        """
        初始化训练指标查看器类。

        参数:
            master: Tkinter的主窗口。
            image_folder: 图像文件夹的路径。
        """
        self.master = master
        self.master.title("Training Metrics Viewer")  # 设置窗口标题
        self.master.configure(bg='gray20')  # 设置窗口背景颜色

        # 配置ttk组件的样式
        style = Style()
        style.theme_use('clam')  # 使用'clam'主题，允许更多的自定义
        style.configure('TScale', background='gray30', troughcolor='gray25', sliderlength=30, borderwidth=1)
        style.configure('Horizontal.TScale', highlightthickness=0)  # 去除高亮边框

        self.image_folder = image_folder
        # 获取所有.png格式的图像文件，并按自然顺序排序
        self.images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')], key=natural_sort_key)
        # 将图像文件加载为PIL的PhotoImage对象
        self.photo_images = [ImageTk.PhotoImage(Image.open(os.path.join(image_folder, img))) for img in self.images]
        self.current_index = 0

        # 创建并显示图像标签
        self.image_label = Label(master, image=self.photo_images[self.current_index], borderwidth=0)
        self.image_label.pack(padx=10, pady=10)

        # 创建ttk.Scale控件，用于在图像之间切换
        self.scale = ttk.Scale(master, from_=0, to=len(self.photo_images) - 1, orient='horizontal',
                               command=self.update_image_from_scale, length=len(self.photo_images) * 10)
        self.scale.pack(fill='x', expand=True, padx=20, pady=20)
        self.scale.set(self.current_index)  # 设置初始滑块位置

    def update_image_from_scale(self, event):
        """
        根据Scale控件的位置更新显示的图像。

        参数:
            event: Scale控件的当前值。
        """
        self.current_index = int(float(event))  # 将Scale返回的浮点数转换为整数
        self.image_label.configure(image=self.photo_images[self.current_index])  # 更新图像标签

def launch_metrics_viewer(image_folder):
    """
    启动训练指标查看器。

    参数:
        image_folder: 包含图像的文件夹路径。
    """
    root = tk.Tk()  # 创建主窗口
    app = MetricsViewer(root, image_folder)  # 初始化MetricsViewer对象
    root.mainloop()  # 启动主事件循环
