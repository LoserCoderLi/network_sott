# -------------------------------------老师自定义数据----------------------------------------
# from SOTT_NN import NN_us
# from NetworkInformation import print_network_architecture
# import numpy as np
#
# nn = NN_us()
#
# X = np.array([[0, 0, 1],
#  [0, 1, 1],
#  [1, 0, 1],
#  [1, 1, 1]])
#
# #通过一个矩阵来定义期望输出
# y = np.array([[0, 1, 1, 0]]).T
# #或者写为：np.array([[0],[1],[1],[0]])
#
# nn.compile(loss="binary_crossentropy", learning_rate=0.01, optimizer='adagrad')
#
#
# nn.add_layer(3, 5, )
# nn.add_activation('sigmoid')
# nn.add_layer(5, 1, )
#
#
# nn.fit(X, y, batch_size=1, epochs=10000)
#
# print(nn.forward(X))
# # print(np.where(nn.forward(X) > 0, 1, 0))
# print_network_architecture(nn)
# nn.loss_show()





# # --------------------------------------波士顿房价预测------------------------------------------------------
# from SOTT_NN import NN_us
# from NetworkInformation import print_network_architecture
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # 加载波士顿房价数据
# boston = load_boston()
# X, y = boston.data, boston.target
#
# # 数据标准化~N（0，1）
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# print(type(X_train))
# print(type(y_train))

# # 创建神经网络实例
# nn = NN_us()
#
# # 指定优化器和损失函数
# nn.compile(optimizer='adam', learning_rate=0.01, loss='mse')
#
# # 搭建模型框架
# nn.add_layer(input_dim=X_train.shape[1], output_dim=64, batch_normalize=False)
# nn.add_activation('relu')
# nn.add_layer(input_dim=64, output_dim=32, batch_normalize=False)
# nn.add_activation('relu')
# nn.add_layer(32, 1)
#
# # 训练模型
# nn.fit(X_train, y_train, batch_size=32, epochs=6666)
#
# # 打印预测值
# y_pred = nn.forward(X_test)
# # # print(y_pred)
# # nn.evaluate_metrics(y_pred,y_test)
#
# # 显示对比曲线
# nn.plot_predictions(y_pred, y_test)
#
# # 显示损失曲线
# nn.loss_show()
#
# # 显示网络架构
# print_network_architecture(nn)
# #




# # ----------------------------------------mnist手写数字识别--------------------------------------------
#
# import torch
# from torchvision import datasets, transforms
# import numpy as np
# from SOTT_NN import NN_us
# import os
# from NetworkInformation import print_network_architecture, launch_metrics_viewer
# import time
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
# # 定义 PyTorch 转换
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
#
# # 加载数据集
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
#
# # 限制数据集大小至前5000张
# # 注意：这里假设数据集已经加载且transform为ToTensor()
# X_train = train_dataset.data[:5000].numpy().reshape(-1, 28*28) / 255.0
# y_train = np.eye(10)[train_dataset.targets[:5000].numpy()]
#
# X_test = test_dataset.data[:5000].numpy().reshape(-1, 28*28) / 255.0
# y_test = np.eye(10)[test_dataset.targets[:5000].numpy()]
#
# # 初始化网络
# model = NN_us()
#
# # 编译模型，指定优化器和损失函数
# model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy_withsoftmax')
#
# # 添加网络层
# model.add_layer(input_dim=784, output_dim=100)  # 输入层到隐藏层1
# model.add_activation('relu')                    # ReLU激活函数
# model.add_layer(input_dim=100, output_dim=20)   # 隐藏层1到隐藏层2
# model.add_activation('relu')                    # ReLU激活函数
# model.add_layer(input_dim=20, output_dim=10)    # 隐藏层2到输出层
#
# # 开始计时
# start_time = time.time()
# # 训练模型
# model.fit(train_data=X_train, train_labels=y_train, batch_size=64, epochs=30, save_metrics=['loss', 'accuracy'])
#
# # 结束计时
# end_time = time.time()
#
#
# # 执行模型前向传播以获取测试数据的预测结果
# y_pred = model.forward(X_test)
#
# # 将预测概率转换为类别编号
# # np.argmax 函数沿指定轴返回最大值的索引，这里我们使用它来获取概率最高的类别的索引
# predicted_labels = np.argmax(y_pred, axis=1)
#
# # 将真实标签也转换为类别编号，因为它们当前是one-hot编码的
# true_labels = np.argmax(y_test, axis=1)
#
# # 计算准确率：正确预测的数量除以总数量
# accuracy = np.mean(predicted_labels == true_labels)
#
# # print(f"Accuracy: {accuracy * 100:.2f}%")
#
# Result = model.evaluate_metrics(y_test, y_pred)
#
# print("SOTT_NN评估指标:")
# print(f"运行时间: {end_time - start_time:.2f} seconds.")
# print(f"准确率: {Result[0]:.4f}")
# print(f"精确率: {Result[1]:.4f}")
# print(f"召回率: {Result[2]:.4f}")
# print(f"F1值: {Result[3]:.4f}")
#
# model.loss_show(save_metrics=['loss', 'accuracy'])
# launch_metrics_viewer("D:\\Awork\\myProjectSum\\pythonProject\\Stride_of_the_Titan_copy\\Stride_of_the_Titan\\training_plots_withnone\\accuracy")
#
# print_network_architecture(model)

# # -------------------------------------------乳腺癌识别---------------------------------------------
# from SOTT_NN import NN_us
# from NetworkInformation import print_network_architecture
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # 加载数据
# data = load_breast_cancer()
# X, y = data.data, data.target
#
# # 划分数据集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 数据标准化
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# # print(X_train.shape)
# print(type(X_train))
# print(X_train)
# print(type(y_train))
# print(y_train)
#
# nn = NN_us()
#
# nn.compile(optimizer='adam', learning_rate=0.001, loss='binary_crossentropy')
#
# nn.add_layer(input_dim= X_train.shape[1], output_dim=16)
# nn.add_activation('relu')
# nn.add_layer(16,1)
# nn.add_activation('sigmoid')
#
# nn.fit(X_train, y_train, batch_size=1, epochs=1000)
#
# y_pred = nn.forward(X_test)
# print(type(y_pred))
# print(type(y_test))
# nn.evaluate_metrics(y_test, y_pred)
#
# nn.loss_show()
#
# print_network_architecture(nn)

from torchvision import datasets, transforms
import numpy as np
from SOTT_NN import NN_us  # 确保 NN_us 包含新增的层和方法
import os
from utils.SAL import save_model_by_layer
import time
from data_tools.Dataset import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 定义 PyTorch 转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)


# # 转换数据以适应卷积网络输入和进行One-Hot编码
# X_train = train_dataset.data.unsqueeze(1).float() / 255.0  # 添加一个维度，适应卷积层输入，并归一化数据
# y_train = np.eye(10)[train_dataset.targets.numpy()]  # 对标签进行One-Hot编码
#
# X_test = test_dataset.data.unsqueeze(1).float() / 255.0
# y_test = np.eye(10)[test_dataset.targets.numpy()]

# 归一化的常数 - 例如MNIST的值
# mean = 0.1307
# std = 0.3081

# 限制数据集大小至前5000张
X_train = train_dataset.data[:5000].unsqueeze(1).float() / 255.0  # 调整形状适合卷积层
# X_train = (X_train - mean) / std  # 归一化和减去均值

y_train = np.eye(10)[train_dataset.targets[:5000].numpy()]

# 注意：这里假设数据集已经加载且transform为ToTensor()
# X_train = train_dataset.data[:5000].numpy().reshape(-1, 28*28) / 255.0
# y_train = np.eye(10)[train_dataset.targets[:5000].numpy()]

X_test = test_dataset.data[:5000].unsqueeze(1).float() / 255.0
# X_test = (X_test - mean) / std  # 归一化和减去均值

y_test = np.eye(10)[test_dataset.targets[:5000].numpy()]

# X_test = test_dataset.data[:5000].numpy().reshape(-1, 28*28) / 255.0
# y_test = np.eye(10)[test_dataset.targets[:5000].numpy()]

data_load = DataLoader(X_train, y_train, 64, shuffle=True)
data_load_test = DataLoader(X_test, y_test, batch_size=1)

# model = NN_us()
# model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy_withsoftmax')
# #
# model.add_conv(1,3,5,1,2,)
# # model.add_batch_norm(16)
# model.add_activation('relu')
# model.add_max_pool(2,2)
#
# model.add_conv(3,3,3,1,1)
# # model.add_batch_norm(32)
# model.add_activation('relu')
# # model.add_residual_block(16, 32, 3)
# model.add_max_pool(2,2)
#
# # model.add_conv(32,64,3,1,1)
# # # model.add_batch_norm(64)
# # model.add_activation('relu')
#
# model.add_flatten()
# model.add_layer(147, 120)
# model.add_activation('relu')
# model.add_layer(120, 10)
# model.add_activation('relu')
# model.add_layer(128, 10)

# # # 添加网络层
# model.add_layer(input_dim=784, output_dim=100)  # 输入层到隐藏层1
# model.add_activation('relu')                    # ReLU激活函数
# model.add_layer(input_dim=100, output_dim=20)   # 隐藏层1到隐藏层2
# model.add_activation('relu')                    # ReLU激活函数
# model.add_layer(input_dim=20, output_dim=10)    # 隐藏层2到输出层

#
# # 初始化网络
model = NN_us()

# 编译模型，指定优化器和损失函数   # steps = epoch * (总数据量 / batch_size) * 全连接层的数量 * 2（chatgpt）
model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy_withsoftmax', init_method='he')

# 添加卷积层和池化层
model.add_conv(input_channels=1, output_channels=16, kernel_size=3, padding=1)
# model.add_batch_norm(16)
model.add_activation('relu')
# model.add_dropout(0.25)
model.add_max_pool(pool_size=2, stride=2)

model.add_conv(input_channels=16, output_channels=32, kernel_size=3, padding=1)
# # model.add_batch_norm(32)
model.add_activation('relu')
# # model.add_dropout(0.25)
model.add_max_pool(pool_size=2, stride=2)
# model.add_residual_block(16, 32, 1)
# model.add_max_pool(pool_size=2, stride=2)

model.add_conv(32, 64, 3)
# model.add_batch_norm(64)
model.add_activation('relu')
# model.add_max_pool(2, 2)


# 添加一个展平操作
model.add_flatten()

# 添加全连接层
model.add_layer(input_dim=1600, output_dim=512)  # 输入维度取决于卷积和池化层的输出
model.add_activation('relu')
model.add_dropout(0.5)
model.add_layer(input_dim=512, output_dim=10)
# model.add_activation('relu')
# model.add_dropout(0.5)
# model.add_layer(input_dim=256, output_dim=128)
# model.add_activation('relu')
# model.add_layer(input_dim=128, output_dim=10)

#
# # 开始计时
# start_time = time.time()
#
# # 训练模型
# model.fit(train_data=X_train, train_labels=y_train, batch_size=64, epochs=10, save_metrics=['loss', 'accuracy'])
# # 初始化网络
# model = NN_us()
#
# # 编译模型，指定优化器和损失函数
# model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy_withsoftmax')
#
# # 添加卷积层和池化层
# model.add_conv(input_channels=1, output_channels=16, kernel_size=3, padding=1)
# # model.add_batch_norm(16)
# model.add_activation('relu')
# # model.add_dropout(0.25)
# model.add_max_pool(pool_size=2, stride=2)
#
# # 添加一个残差块
# model.add_residual_block(16, 32, stride=1)
# # model.add_batch_norm(32)
# # model.add_dropout(0.25)
# model.add_max_pool(pool_size=2, stride=2)
#
# model.add_residual_block(32, 64, stride=1)
# model.add_activation('relu')
#
# # 添加一个展平操作
# model.add_flatten()
#
# # 添加全连接层
# model.add_layer(input_dim=3136, output_dim=128)  # 输入维度取决于卷积和池化层的输出
# model.add_activation('relu')
# model.add_layer(input_dim=128, output_dim=10)
# #
# #

# 开始计时

start_time = time.time()

# # 训练模型
# model.fit(train_data=X_train, train_labels=y_train, batch_size=256, epochs=10, save_metrics=['loss', 'accuracy'])
# 训练模型
model.fit(data_load, data_load_test, epochs=30)

# 结束计时
end_time = time.time()

save_model_by_layer(model, 'mnist')

# 执行模型前向传播以获取测试数据的预测结果
y_pred = model.forward(X_test)

# 将预测概率转换为类别编号
predicted_labels = np.argmax(y_pred, axis=1)

# 将真实标签也转换为类别编号
true_labels = np.argmax(y_test, axis=1)

# 计算准确率
accuracy = np.mean(predicted_labels == true_labels)

Result = model.evaluate_metrics(y_test, y_pred)

print("SOTT_NN评估指标:")
print(f"运行时间: {end_time - start_time:.2f} seconds.")
print(f"准确率: {Result[0]:.4f}")
print(f"精确率: {Result[1]:.4f}")
print(f"召回率: {Result[2]:.4f}")
print(f"F1值: {Result[3]:.4f}")

model.loss_show(save_metrics=['loss', 'accuracy'])

# 以下函数假设已经在其他模块中定义
from utils.NetworkInformation import print_network_architecture, launch_metrics_viewer
launch_metrics_viewer("D:\\Awork\\myProjectSum\\pythonProject\\Stride_of_the_Titan_copy\\Stride_of_the_Titan\\training_plots\\accuracy")
print_network_architecture(model)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#
#
# # 定义生成器
# class Generator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Generator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.LeakyReLU(0.2),
#             nn.Linear(128, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, output_dim),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
# # 定义判别器
# class Discriminator(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
# # 功能函数：显示生成的图像
# def show_images(images, num_images):
#     images = images.view(images.size(0), 28, 28)
#     images = images.detach().numpy()
#     fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
#     for i, ax in enumerate(axes):
#         ax.imshow(images[i], cmap='gray')
#         ax.axis('off')
#     plt.show()
#
# # 初始化网络和优化器
# generator = Generator(100, 28*28)
# discriminator = Discriminator(28*28)
# g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
# d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
#
# # 数据加载和训练设置
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#
# # 训练过程
# num_epochs = 30
# for epoch in range(num_epochs):
#     for i, (images, _) in enumerate(train_loader):
#         # 训练判别器
#         images = images.view(images.size(0), -1)
#         real_labels = torch.ones(images.size(0), 1)
#         fake_labels = torch.zeros(images.size(0), 1)
#
#         outputs = discriminator(images)
#         d_loss_real = nn.BCELoss()(outputs, real_labels)
#         real_score = outputs
#
#         z = torch.randn(images.size(0), 100)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images)
#         d_loss_fake = nn.BCELoss()(outputs, fake_labels)
#         fake_score = outputs
#
#         d_loss = d_loss_real + d_loss_fake
#         d_optimizer.zero_grad()
#         g_optimizer.zero_grad()
#         d_loss.backward()
#         d_optimizer.step()
#
#         # 训练生成器
#         z = torch.randn(images.size(0), 100)
#         fake_images = generator(z)
#         outputs = discriminator(fake_images)
#         g_loss = nn.BCELoss()(outputs, real_labels)
#
#         d_optimizer.zero_grad()
#         g_optimizer.zero_grad()
#         g_loss.backward()
#         g_optimizer.step()
#
#         if (i+1) % 400 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')
#
#     # 在每个epoch结束时显示生成的图像
#     show_images(fake_images, num_images=6)
