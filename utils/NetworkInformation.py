from layers.Linear_Act import FullyConnectedLayer, Activation
import tkinter as tk
from tkinter import ttk, Label
from tkinter.ttk import Style
from PIL import Image, ImageTk
import os
import re
from layers.Conv_Pool import ConvolutionLayer, MaxPoolingLayer, FlattenLayer
from layers.BatchNorm import BatchNormalization
from layers.Residual import ResidualBlock


def print_network_architecture(neural_network):
    print("网络结构:")
    for i, layer in enumerate(neural_network.layers):
        layer_type = type(layer).__name__
        print(f"Layer {i + 1}: {layer_type}")
        if isinstance(layer, FullyConnectedLayer):
            print(f"   Input Dimension: {layer.weights.shape[1]}")
            print(f"   Output Dimension: {layer.weights.shape[0]}")
        elif isinstance(layer, ConvolutionLayer):
            print(f"   Input Channels: {layer.input_channels}")
            print(f"   Output Channels: {layer.output_channels}")
            print(f"   Kernel Size: {layer.kernel_size}x{layer.kernel_size}")
            print(f"   Stride: {layer.stride}")
            print(f"   Padding: {layer.padding}")
        elif isinstance(layer, MaxPoolingLayer):
            print(f"   Pool Size: {layer.pool_size}x{layer.pool_size}")
            print(f"   Stride: {layer.stride}")
        elif isinstance(layer, Activation):
            print(f"   Activation Function: {layer.activation}")
        elif isinstance(layer, BatchNormalization):
            print(f"   Number of Features: {layer.gamma.shape[1]}")
            print(f"   Epsilon: {layer.eps}")
            print(f"   Momentum: {layer.momentum}")
        elif isinstance(layer, FlattenLayer):
            print("   Flattens the input.")
        elif isinstance(layer, ResidualBlock):
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
    """提取文本中的数字用于排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class MetricsViewer:
    def __init__(self, master, image_folder):
        self.master = master
        self.master.title("Training Metrics Viewer")
        self.master.configure(bg='gray20')  # Set background color of the window

        # Set up the style for the ttk widgets
        style = Style()
        style.theme_use('clam')  # Use the 'clam' theme as it allows more customization
        style.configure('TScale', background='gray30', troughcolor='gray25', sliderlength=30, borderwidth=1)
        style.configure('Horizontal.TScale', highlightthickness=0)  # Remove highlight borders

        self.image_folder = image_folder
        self.images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')], key=natural_sort_key)
        self.photo_images = [ImageTk.PhotoImage(Image.open(os.path.join(image_folder, img))) for img in self.images]
        self.current_index = 0

        self.image_label = Label(master, image=self.photo_images[self.current_index], borderwidth=0)
        self.image_label.pack(padx=10, pady=10)

        # Create a ttk.Scale instead of tkinter Scale
        self.scale = ttk.Scale(master, from_=0, to=len(self.photo_images) - 1, orient='horizontal',
                               command=self.update_image_from_scale, length=len(self.photo_images) * 10)
        self.scale.pack(fill='x', expand=True, padx=20, pady=20)
        self.scale.set(self.current_index)

    def update_image_from_scale(self, event):
        self.current_index = int(float(event))  # ttk Scale returns a float
        self.image_label.configure(image=self.photo_images[self.current_index])

def launch_metrics_viewer(image_folder):
    """启动图像浏览器的函数"""
    root = tk.Tk()
    app = MetricsViewer(root, image_folder)
    root.mainloop()