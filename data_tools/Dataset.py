import numpy as np
from PIL import Image
import threading
import queue
from tqdm import tqdm
import csv
from PIL import Image
from multiprocessing import Queue, Process


class DataLoader:
    def __init__(self, data, labels, batch_size=32, shuffle=True, augmenter=None):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = augmenter
        self.indexes = np.arange(len(self.data))
        self.lock = threading.Lock()

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.n = 0
        return self

    def augment_batch(self, batch_data):
        if self.augmenter:
            if batch_data.ndim == 4 and batch_data.shape[1] == 1:
                batch_data = batch_data.squeeze(1)  # 从 (batch_size, 1, height, width) 转换为 (batch_size, height, width)
            augmented_data = np.array(
                [np.array(self.augmenter.augment(Image.fromarray(image.astype(np.uint8)))) for image in batch_data]
            )
            # 恢复维度到 (batch_size, 1, height, width)
            augmented_data = augmented_data[:, np.newaxis, :, :]  # 从 (batch_size, height, width) 转换为 (batch_size, 1, height, width)
            return augmented_data
        return batch_data

    def __next__(self):
        with self.lock:
            if self.n < len(self.data):
                start = self.n
                end = self.n + self.batch_size
                self.n += self.batch_size
                batch_indexes = self.indexes[start:end]
                batch_data = self.data[batch_indexes]
                batch_labels = self.labels[batch_indexes]

                return self.augment_batch(batch_data), batch_labels
            else:
                raise StopIteration()

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
