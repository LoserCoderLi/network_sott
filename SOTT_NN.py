from torchvision import transforms
import os
from training.Loss import LossFunctions
from layers.Linear_Act import FullyConnectedLayer, Activation
from training.Optimizer import Optimizers, LearningRateScheduler
from layers.Conv_Pool import ConvolutionLayer, MaxPoolingLayer, FlattenLayer
from layers.BatchNorm import BatchNormalization
from layers.Drop import Dropout
from layers.Residual import ResidualBlock
from data_tools.Dataset import DataLoader
from torchvision import datasets
import numpy as np
from tqdm import tqdm  # 导入tqdm
import matplotlib.pyplot as plt
import csv
import h5py

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置 matplotlib 的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体，例如使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题


class MetricsHistory:
    def __init__(self, save_dir='./training_plots', save_metrics=None):
        self.metrics_data = {
            'loss': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }
        self.save_dir = save_dir
        self.save_metrics = save_metrics if save_metrics else self.metrics_data.keys()  # 如果没有指定就保存所有指标
        self.current_plot_paths = {}
        os.makedirs(self.save_dir, exist_ok=True)
        for metric in self.metrics_data.keys():
            os.makedirs(os.path.join(self.save_dir, metric), exist_ok=True)

    def add_metrics(self, loss, precision, recall, f1, accuracy):
        self.metrics_data['loss'].append(loss)
        self.metrics_data['precision'].append(precision)
        self.metrics_data['recall'].append(recall)
        self.metrics_data['f1_score'].append(f1)
        self.metrics_data['accuracy'].append(accuracy)

        print("add_metrics(self, loss, precision, recall, f1, accuracy):self, ", loss, precision, recall, f1, accuracy)

    def smooth_curve(self, points, factor=0.8):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def plot(self):
        epochs = range(1, len(self.metrics_data['loss']) + 1)
        for key, values in self.metrics_data.items():
            if key in self.save_metrics:  # 只有在用户想要保存的指标列表中的指标才进行绘图和保存
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, self.smooth_curve(values), label=f'{key.title()}')
                plt.xlabel('Epoch')
                plt.ylabel(f'{key.title()} Value')
                plt.title(f'{key.title()} Metrics Over Epochs')
                plt.legend(loc='best')
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plot_path = os.path.join(self.save_dir, key, f'{key}_metrics_epoch_{len(epochs)}.png')
                plt.savefig(plot_path)
                plt.close()  # 关闭图表以防显示
                self.current_plot_paths[key] = plot_path

    def show(self, metric):
        if metric in self.current_plot_paths:
            img = plt.imread(self.current_plot_paths[metric])
            plt.imshow(img)
            plt.axis('off')  # 隐藏坐标轴
            plt.show()



class NN_us:
    def __init__(self):
        self.layers_inf = []
        self.layers = []
        self._optimizer = None
        self.metrics_history = MetricsHistory()  # 添加损失历史记录器
        self.learning_rate = 0
        self.optimizer_name = None
        self.loss_name = None
        self.is_compiled = False  # 初始状态，模型未编译
        self.init_method = 'he'
        # 定义 PyTorch 转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

        # 注意：这里假设数据集已经加载且transform为ToTensor()
        X_train = train_dataset.data[:5000].unsqueeze(1).float() / 255.0  # 调整形状适合卷积层

        y_train = np.eye(10)[train_dataset.targets[:5000].numpy()]

        X_test = test_dataset.data[:5000].unsqueeze(1).float() / 255.0
        y_test = np.eye(10)[test_dataset.targets[:5000].numpy()]

        self.data_load_mnist = DataLoader(X_train, y_train, 64, shuffle=True)
        self.data_load_test_mnist = DataLoader(X_test, y_test, batch_size=1)

        # # 载入本地文件夹中的图片
        # fire_images, fire_labels = Data_Power.load_images_from_folder(r'D:\Awork\Data\data\train\Fire', 1,
        #                                                               target_size=(256, 256))
        # non_fire_images, non_fire_labels = Data_Power.load_images_from_folder(r'D:\Awork\Data\data\train\No Fire', 0,
        #                                                                       target_size=(256, 256))
        #
        # # 将加载的图片和标签转换为NumPy数组
        # fire_images = np.array(fire_images)
        # fire_labels = np.array(fire_labels)
        # non_fire_images = np.array(non_fire_images)
        # non_fire_labels = np.array(non_fire_labels)
        #
        # # 合并数据和标签
        # train_data = np.concatenate((fire_images, non_fire_images), axis=0)
        # train_labels = np.concatenate((fire_labels, non_fire_labels), axis=0)
        #
        # # 则展平后的大小应为 (num_samples, 256*256*3)
        # train_data = train_data.reshape(train_data.shape[0], -1)
        # # 数据归一化
        # train_data = train_data / 255.0
        #
        # train_labels = np.eye(2)[train_labels]
        #
        # # 定义数据加载器
        # self.data_load_fire = DataLoader(train_data, train_labels, batch_size=16)

    def get_layers_inf(self):
        return [str(layer) for layer in self.layers_inf]  # 将每个层对象转换为字符串描述

    def get_layers(self):
        return self.layers

    def compile(self, optimizer, learning_rate, loss, init_method='he', lr_schedule='normal', steps = None):
        self.lr_schedule_name = lr_schedule
        if lr_schedule != 'normal' :
            if steps == None:
                self.steps = [1000, 10000]
            else:self.steps = steps

            if self.steps[0] >= self.steps[1]:
                raise Exception(f'steps[0]的数值应该小于stpes[1]的数值')

            self.lr_schedule = LearningRateScheduler(initial_lr=0.01, schedule_type=lr_schedule, warmup_steps=self.steps[0], total_steps=self.steps[1])
        else:
            self.lr_schedule = LearningRateScheduler(initial_lr=0.01, schedule_type=lr_schedule)

        self.learning_rate = learning_rate
        self.init_method = init_method
        # self.loss = loss

        ## 优化器
        if optimizer == 'sgd':
            self._optimizer = Optimizers.SGD(learning_rate, scheduler=self.lr_schedule)
            self.optimizer_name = 'sgd'
        elif optimizer == 'adam':
            self._optimizer = Optimizers.Adam(learning_rate, scheduler=self.lr_schedule)
            self.optimizer_name = 'adam'
        elif optimizer == 'rms':
            self._optimizer = Optimizers.RMSprop(learning_rate, scheduler=self.lr_schedule)
            self.optimizer_name = 'rms'
        elif optimizer == 'adagrad':
            self._optimizer = Optimizers.Adagrad(learning_rate, scheduler=self.lr_schedule)
            self.optimizer_name = 'adagrad'
        else:
            raise Exception(f'Unknown optimizer: {optimizer}')

        ## 损失函数
        if loss == 'categorical_crossentropy':
            self.loss = LossFunctions.CategoricalCrossEntropy()
            self.loss_name = 'categorical_crossentropy'
        elif loss == 'categorical_crossentropy_withsoftmax':
            self.loss = LossFunctions.CrossEntropyLossWithSoftmax()
            self.loss_name = 'categorical_crossentropy_withsoftmax'
        elif loss == 'binary_crossentropy':
            self.loss = LossFunctions.BinaryCrossEntropy()
            self.loss_name = 'binary_crossentropy'
        elif loss == 'mse':
            self.loss = LossFunctions.MeanSquaredError()
            self.loss_name = 'mse'
        elif loss == 'huber':
            self.loss = LossFunctions.HuberLoss()
            self.loss_name = 'hl'
        elif loss == 'log_cosh':
            self.loss = LossFunctions.LogCoshLoss()
            self.loss_name = 'log_cosh'
        else:
            raise Exception(f'Unknown loss function: {loss}')

        self.is_compiled = True  # 编译模型后更新状态
        # model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy_withsoftmax',
        #               init_method='he', lr_schedule='rate_warmup', steps=[1000, 10000])
        # self.layers_inf.append(f"Compile: optimizer={optimizer}, learning_rate={learning_rate}, loss={loss}")
        self.layers_inf.append("model = NN_us()")
        self.layers_inf.append(f"model.compile(optimizer='{optimizer}', learning_rate={learning_rate}, loss='{loss}',init_method='{init_method}', lr_schedule='{lr_schedule}', steps={steps})")

    def add_conv(self, input_channels, output_channels, kernel_size, stride=1, padding=0, conv = False, name = 'conv1'):
        print(self.init_method)
        if self._optimizer is None:
            raise Exception('You must compile the model before adding layers')

        layer = ConvolutionLayer(input_channels=input_channels, output_channels=output_channels, kernel_size=kernel_size,optimizer=self._optimizer, stride=stride, padding=padding, Conv=conv, init_method=self.init_method, name=name)
        self.layers.append(layer)
        # layer_inf = f"Conv Layer: input_channels={input_channels}, output_channels={output_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, conv={conv}"
        layer_inf = f"model.add_conv(input_channels={input_channels}, output_channels={output_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})"
        # model.add_conv(input_channels=1, output_channels=16, kernel_size=3, padding=1)

        self.layers_inf.append(layer_inf)
        print("ConvolutionLayer", input_channels, output_channels, kernel_size, stride, padding, len(self.layers), len(self.layers_inf))

    # def add_convn(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
    #     layer = ConvolutionLayerN(input_channels, output_channels, kernel_size, stride, padding)
    #     self.layers.append(layer)

    def add_max_pool(self, pool_size, stride, name = 'pool'):
        layer = MaxPoolingLayer(pool_size, stride, name=name)
        self.layers.append(layer)
        # layer_inf = f"Max Pool: pool_size={pool_size}, stride={stride}"
        layer_inf = f"model.add_max_pool(pool_size={pool_size}, stride={stride})"
        # model.add_max_pool(pool_size=2, stride=2)

        self.layers_inf.append(layer_inf)
        print("MaxPoolingLayer", pool_size, stride,len(self.layers), len(self.layers_inf))

    def add_batch_norm(self, num_features):
        bn_layer = BatchNormalization(num_features)
        self.layers.append(bn_layer)
        # layer = f"Batch Norm: num_features={num_features}"
        layer = f"model.add_batch_norm({num_features})"
        # # model.add_batch_norm(32)

        self.layers_inf.append(layer)
        print("BatchNormalization", num_features,len(self.layers), len(self.layers_inf))

    def add_flatten(self, name = 'flatten'):
        layer = FlattenLayer(name=name)
        self.layers.append(layer)
        # layer = "Flatten"
        layer = "model.add_flatten()"
        # model.add_flatten()

        self.layers_inf.append(layer)
        print("FlattenLayer",len(self.layers), len(self.layers_inf))

    def add_layer(self, input_dim, output_dim, name = 'fc'):
        if self._optimizer is None:
            raise Exception('You must compile the model before adding layers')
        print(self.init_method)
        layer = FullyConnectedLayer(input_dim, output_dim, optimizer=self._optimizer, init_method=self.init_method, name=name)
        self.layers.append(layer)
        # layer = f"Layer: input_dim={input_dim}, output_dim={output_dim}"
        layer = f"model.add_layer(input_dim={input_dim}, output_dim={output_dim})"
        # model.add_layer(input_dim=1600, output_dim=512)  # 输入维度取决于卷积和池化层的输出

        self.layers_inf.append(layer)
        print("FullyConnectedLayer",input_dim, output_dim,type(input_dim), self._optimizer, self.init_method, len(self.layers), len(self.layers_inf))

    def add_activation(self, str):
        layer = Activation(str)
        self.layers.append(layer)
        # layer = f"Activation: {str}"
        layer = f"model.add_activation('{str}')"
        # model.add_activation('relu')

        self.layers_inf.append(layer)
        print("Activation",str,len(self.layers), len(self.layers_inf))

    def add_residual_block(self, input_channels, output_channels,stride=1, use_batch_norm = False):

        res_block = ResidualBlock(input_channels=input_channels, output_channels=output_channels, stride=stride, use_batch_norm=use_batch_norm, opm=self._optimizer)
        self.layers.append(res_block)

        # layer = f"Residual Block: input_channels={input_channels}, output_channels={output_channels}, stride={stride}, use_batch_norm={use_batch_norm}"
        layer = f"model.add_residual_block(input_channels={input_channels}, output_channels={output_channels}, stride={stride})"
        # model.add_residual_block(16, 32, stride=1)

        self.layers_inf.append(layer)
        print("Residual Block",input_channels, output_channels, stride,len(self.layers), len(self.layers_inf))

    def add_dropout(self, p=0.5):
        dropout_layer = Dropout(p)
        self.layers.append(dropout_layer)
        # layer = f"Dropout: p={p}"
        layer = f"model.add_dropout(p={p})"
        # # model.add_dropout(0.25)

        self.layers_inf.append(layer)
        print("Dropout",Dropout,len(self.layers), len(self.layers_inf))

    def forward(self, X, is_training=True):
        for layer in self.layers:
            # 检查层是否接受is_training参数
            # print(f'Forward layer: {layer}, input shape: {X.shape}')
            if isinstance(layer, (BatchNormalization, Dropout)):
                X = layer.forward(X, is_training)
            else:
                X = layer.forward(X)
            # print(f'Output shape: {X.shape}')
        return X

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            # print(f'Backward layer: {layer}, grad_output shape: {grad_output.shape}')
            grad_output = layer.backward(grad_output)  # 确保这一行正确处理
            # print(f'Updated grad_output shape: {grad_output.shape}')
        return grad_output

    def Visualization_display(self, count, sumCount):
        print("SOTT正在疯狂运作中，老爷您休息会！！-----------------(" + str(count+1) + "/" + str(sumCount) + ")")

    def fit(self, data_loader = None, data_loader_test = None, epochs=100, save_metrics=None, save_dir='./training_plots'):

        if data_loader is None:
            data_loader = self.data_load_mnist
        if data_loader_test is None:
            data_loader_test = self.data_load_test_mnist

        if self.lr_schedule_name != 'normal':
            if self.steps[1] < epochs * data_loader.batch_size:
                raise Exception(
                    '当前你的训练次数大于steps[1](训练的总步数)，当训练的次数大于steps[1](训练的总步数)时，之后的训练中，学习率都为1，无法训练')

        self.metrics_history = MetricsHistory(save_dir=save_dir, save_metrics=save_metrics)
        print("正在训练中，喝点水休息一下吧。")

        # 初始化CSV文件
        csv_file = 'training_metrics.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1'])

        for epoch in range(epochs):
            loss_sum = 0
            total_batches = len(data_loader)

            for batch_data, batch_labels in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                # 向前传递
                preds = self.forward(batch_data)

                # 计算损失函数
                loss = np.mean(self.loss.loss(batch_labels, preds))
                loss_sum += loss

                # 向后传递
                grads = self.loss.grad(batch_labels, preds)
                self.backward(grads)

                # 更新参数
                for layer in self.layers:
                    if isinstance(layer, FullyConnectedLayer):
                        print("Full")
                        layer.update()
                    # if isinstance(layer, BatchNormalization):
                    #     layer.update_params()
                    if isinstance(layer, ConvolutionLayer):
                        if layer.Conv:
                            print("conv")
                            layer.update()


            # 计算整个训练集的性能指标
            if data_loader_test == None:
                train_data, train_labels = data_loader.data, data_loader.labels  # 假设 DataLoader 有属性 dataset
                train_pred = self.forward(train_data)
                accuracy, precision, recall, f1 = self.evaluate_metrics(train_labels, train_pred)
            else:
                test_data, test_labels = data_loader_test.data, data_loader_test.labels  # 假设 DataLoader 有属性 dataset
                test_pred = self.forward(test_data)
                accuracy, precision, recall, f1 = self.evaluate_metrics(test_labels, test_pred)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                print("writer.writerow[epoch + 1, loss_sum / total_batches, accuracy, precision, recall, f1]", epoch + 1, loss_sum / total_batches, accuracy, precision, recall, f1)
                writer.writerow([epoch + 1, loss_sum / total_batches, accuracy, precision, recall, f1])

            # print("metrics_history.add_metrics(loss_sum / total_batches, precision, recall, f1, accuracy)", loss_sum / total_batches, precision, recall, f1, accuracy)
            # self.metrics_history.add_metrics(loss_sum / total_batches, precision, recall, f1, accuracy)
            # self.metrics_history.plot()

    def load_vgg16_weights(self, weight_file_path, import_mode='all'):
        """
        Load VGG16 weights into the neural network.

        Parameters:
        - weight_file_path: str, path to the VGG16 weights file.
        - import_mode: str, 'all' to import all layers, 'conv_pool' to import only conv and pool layers.
        """
        with h5py.File(weight_file_path, 'r') as f:
            if import_mode == 'all':
                # Import all weights
                for layer in self.layers:
                    if layer.name in f:
                        g = f[layer.name]
                        weights = [np.array(g[p]) for p in g]
                        layer.set_weights(weights)
            elif import_mode == 'conv_pool':
                # Import only convolutional and pooling layers
                for layer in self.layers:
                    if 'conv' in layer.name or 'pool' in layer.name:
                        if layer.name in f:
                            g = f[layer.name]
                            weights = [np.array(g[p]) for p in g]
                            layer.set_weights(weights)
            else:
                raise ValueError("import_mode should be 'all' or 'conv_pool'")

    def evaluate_metrics(self, y_true, y_pred, average='macro'):

        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape)
        print("y_true sample:", y_true[:5])
        print("y_pred sample:", y_pred[:5])

        # print("y_true type:", type(y_true))
        # print("y_pred type:", type(y_pred))
        # 检查y_pred是否为一维或二维，如果为一维，则进行转换
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            y_pred_labels = (y_pred > 0.5).astype(int)
        else:
            y_pred_labels = np.argmax(y_pred, axis=1)

        if y_true.ndim > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true

        num_classes = np.max(y_true_labels) + 1
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(len(y_true_labels)):
            try:
                confusion_matrix[y_true_labels[i], y_pred_labels[i]] += 1
            except IndexError as e:
                print(f"IndexError: {e}, y_true_labels[i]: {y_true_labels[i]}, y_pred_labels[i]: {y_pred_labels[i]}")

        true_positives = np.diag(confusion_matrix)
        false_positives = np.sum(confusion_matrix, axis=0) - true_positives
        false_negatives = np.sum(confusion_matrix, axis=1) - true_positives

        # 计算微平均和宏平均
        precision_micro = np.sum(true_positives) / np.sum(true_positives + false_positives)
        recall_micro = np.sum(true_positives) / np.sum(true_positives + false_negatives)
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

        epsilon = 1e-8  # 防止除以零
        precision_macro = np.mean(true_positives / (true_positives + false_positives + epsilon))

        recall_macro = np.mean((true_positives + epsilon) / (true_positives + false_negatives + epsilon))
        f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)

        accuracy = np.mean(y_pred_labels == y_true_labels)

        # 根据选择的平均类型返回结果
        if average == 'micro':
            return accuracy, precision_micro, recall_micro, f1_micro
        else:
            return accuracy, precision_macro, recall_macro, f1_macro
    # def plot_predictions(self, y_pred, y_test, ylabel):
    #     # 使用网络进行预测
    #     y_pred = y_pred.flatten()  # 确保 y_pred 是一维数组
    #
    #     # 绘制真实值和预测值的线图
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(y_test, color='blue', label='真实值', alpha=0.6, marker='o')
    #     plt.plot(y_pred, color='red', label='预测值', alpha=0.6, marker='x')
    #     plt.title('真实值 vs 预测值')
    #     plt.xlabel('样本')
    #     plt.ylabel(ylabel)
    #     plt.legend()
    #     plt.show()

    def loss_show(self, save_metrics):
        for i in save_metrics:
            self.metrics_history.show(i)  # 手动绘制损失曲线的方法

        print("训练完成")
