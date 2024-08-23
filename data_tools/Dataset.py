import numpy as np
from PIL import Image
import threading
import queue
from tqdm import tqdm
import csv
from multiprocessing import Queue, Process
 import queue

class DataLoader:
    def __init__(self, data, labels, batch_size=32, shuffle=True, augmenter=None):
        """
        初始化DataLoader对象。

        参数:
            data (numpy.ndarray): 输入数据集，通常为图像数组。
            labels (numpy.ndarray): 输入数据集对应的标签。
            batch_size (int): 每个批次的样本数量，默认为32。
            shuffle (bool): 是否在每个epoch开始时打乱数据顺序，默认为True。
            augmenter (callable): 数据增强函数或对象，如果提供则用于批量增强。
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.indexes = np.arange(len(self.data))
        self.lock = threading.Lock()

    def __iter__(self):
        """
        初始化迭代器。

        如果设置了shuffle为True，则在每次迭代开始时随机打乱索引顺序。
        初始化当前批次计数器self.n为0。
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.n = 0
        return self

    def augment_batch(self, batch_data):
        """
        批量数据增强函数。

        如果提供了数据增强器augmenter，则应用它来增强输入批次数据。
        适用于灰度图像的情况，即输入为(batch_size, 1, height, width)的张量。

        参数:
            batch_data (numpy.ndarray): 输入批次数据。

        返回:
            numpy.ndarray: 增强后的批次数据，维度与输入相同。
        """
        if self.augmenter:
            # 如果输入数据是单通道图像 (batch_size, 1, height, width)
            if batch_data.ndim == 4 and batch_data.shape[1] == 1:
                # 将数据从 (batch_size, 1, height, width) 转换为 (batch_size, height, width)
                batch_data = batch_data.squeeze(1)
            # 对每一张图片进行增强处理
            augmented_data = np.array(
                [np.array(self.augmenter.augment(Image.fromarray(image.astype(np.uint8)))) for image in batch_data]
            )
            # 恢复数据到 (batch_size, 1, height, width) 的格式
            augmented_data = augmented_data[:, np.newaxis, :, :]
            return augmented_data
        return batch_data

    def __next__(self):
        """
        获取下一个批次的数据和标签。

        返回:
            tuple: 包含增强后的批次数据和对应标签。

        抛出:
            StopIteration: 当所有数据都已经被迭代完时。
        """
        with self.lock:
            if self.n < len(self.data):
                # 计算当前批次的起始和结束索引
                start = self.n
                end = self.n + self.batch_size
                self.n += self.batch_size
                batch_indexes = self.indexes[start:end]
                batch_data = self.data[batch_indexes]
                batch_labels = self.labels[batch_indexes]

                # 返回增强后的批次数据和对应的标签
                return self.augment_batch(batch_data), batch_labels
            else:
                raise StopIteration()

    def __len__(self):
        """
        获取总批次数。

        返回:
            int: 总批次数量，根据批次大小计算。
        """
        return (len(self.data) + self.batch_size - 1) // self.batch_size
