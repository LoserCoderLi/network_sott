import numpy as np
import pickle


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.01):
        """
        初始化逻辑回归模型的参数。

        参数:
            learning_rate (float): 学习率，必须大于0。
            num_iterations (int): 训练迭代次数，必须大于0。
            regularization_strength (float): 正则化强度，默认为0.01。

        异常:
            ValueError: 如果学习率或迭代次数小于等于0，抛出异常。
        """
        if learning_rate <= 0 or num_iterations <= 0:
            raise ValueError("Learning rate and number of iterations must be greater than zero.")
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        计算逻辑回归模型的sigmoid函数。

        参数:
            z (np.ndarray): 输入值。

        返回:
            np.ndarray: sigmoid函数的结果。
        """
        z_safe = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_safe))

    def _initialize_parameters(self, n_features):
        """
        初始化模型的权重和偏置。

        参数:
            n_features (int): 特征的数量。
        """
        self.weights = np.random.randn(n_features) * 0.01  # 使用小的随机数初始化权重
        self.bias = 0

    def _compute_loss(self, y, y_hat):
        """
        计算逻辑回归模型的损失，包括L2正则化项。

        参数:
            y (np.ndarray): 实际标签。
            y_hat (np.ndarray): 预测概率。

        返回:
            float: 计算得到的损失值。
        """
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)  # 避免log(0)引发数值问题
        m = y.shape[0]
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
        # 添加L2正则化项
        loss += self.regularization_strength * 0.5 * np.sum(np.square(self.weights))
        return loss

    def fit(self, X, y):
        """
        训练逻辑回归模型。

        参数:
            X (np.ndarray): 训练数据，二维数组。
            y (np.ndarray): 训练标签，一维数组。

        异常:
            ValueError: 如果X不是二维数组，y不是一维数组，或X和y的样本数不一致，抛出异常。
        """
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be a 2D array and y must be a 1D array.")
        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("Number of samples in X and y must be the same.")
        self._initialize_parameters(n_features)

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(z)
            loss = self._compute_loss(y, y_hat)

            dw = (np.dot(X.T, (y_hat - y)) + self.regularization_strength * self.weights) / n_samples
            db = np.sum(y_hat - y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                print(f"Iteration {i}: Loss {loss}")

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
            X (np.ndarray): 测试数据，二维数组。

        返回:
            np.ndarray: 预测的标签。

        异常:
            ValueError: 如果X不是二维数组，抛出异常。
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        z = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(z)
        return np.where(y_hat > 0.5, 1, 0)

    def save(self, filename):
        """
        保存训练好的模型到文件。

        参数:
            filename (str): 保存模型的文件名。
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        从文件中加载训练好的模型。

        参数:
            filename (str): 模型文件的文件名。

        返回:
            LogisticRegression: 加载的逻辑回归模型。
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        """
        初始化决策树分类器。

        参数:
            max_depth (int, 可选): 决策树的最大深度。如果为None，则树的深度没有限制。
        """
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def _calculate_entropy(self, y):
        """
        计算给定标签集合的熵。

        参数:
            y (numpy.ndarray): 标签数组。

        返回:
            float: 熵的值。
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _best_split(self, X, y):
        """
        寻找最佳分割特征和分割点。

        参数:
            X (numpy.ndarray): 特征矩阵。
            y (numpy.ndarray): 标签数组。

        返回:
            tuple: 最佳分割特征的索引，分割阈值，以及该分割的增益。
        """
        best_feature, best_value, best_gain = None, None, -1
        n_features = X.shape[1]
        parent_entropy = self._calculate_entropy(y)
        n = len(y)

        # 如果还没有初始化，则初始化特征重要性数组
        if self.feature_importances_ is None:
            self.feature_importances_ = np.zeros(n_features)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_entropy = self._calculate_entropy(y[left_mask])
                right_entropy = self._calculate_entropy(y[right_mask])
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = threshold

            # 更新特征重要性
            if best_feature is not None:
                self.feature_importances_[best_feature] += gain

        return best_feature, best_value, best_gain

    def _build_tree(self, X, y, depth=0):
        """
        递归构建决策树。

        参数:
            X (numpy.ndarray): 特征矩阵。
            y (numpy.ndarray): 标签数组。
            depth (int): 当前递归的深度。

        返回:
            dict: 决策树的节点。
        """
        num_samples, num_features = X.shape
        if num_samples == 0:
            return None
        if len(np.unique(y)) == 1:
            return {'label': y[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return {'label': np.bincount(y).argmax()}

        feature, value, gain = self._best_split(X, y)
        if gain == -1:
            return {'label': np.bincount(y).argmax()}

        left_indices = X[:, feature] <= value
        right_indices = X[:, feature] > value
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': feature, 'value': value, 'left': left_subtree, 'right': right_subtree}

    def fit(self, X, y):
        """
        训练模型并重置/初始化特征重要性评分。

        参数:
            X (numpy.ndarray): 特征矩阵。
            y (numpy.ndarray): 标签数组。
        """
        self.feature_importances_ = np.zeros(X.shape[1])  # 重置特征重要性
        self.tree = self._build_tree(X, y, 0)
        self.feature_importances_ /= np.sum(self.feature_importances_)  # 归一化特征重要性

    def predict(self, X):
        """
        使用训练好的决策树进行预测。

        参数:
            X (numpy.ndarray): 特征矩阵。

        返回:
            numpy.ndarray: 预测的标签数组。
        """
        results = [self._predict_one(x, self.tree) for x in X]
        return np.array(results)

    def _predict_one(self, x, node):
        """
        递归预测单个样本的标签。

        参数:
            x (numpy.ndarray): 单个样本的特征向量。
            node (dict): 决策树的当前节点。

        返回:
            int: 预测的标签。
        """
        while node.get('feature') is not None:
            if x[node['feature']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
        return node['label']

    def get_feature_importances(self):
        """
        返回特征重要性评分。

        返回:
            numpy.ndarray: 特征重要性数组。
        """
        if self.feature_importances_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.feature_importances_

    def save(self, filename):
        """
        将决策树模型保存到文件。

        参数:
            filename (str): 保存模型的文件名。
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        从文件加载决策树模型。

        参数:
            filename (str): 从哪个文件加载模型。

        返回:
            DecisionTreeClassifier: 加载的模型。
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)


class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        初始化K近邻(KNN)分类器。

        :param k: 邻居的数量，即选择最近的k个样本进行投票。
        :param distance_metric: 用于计算样本之间距离的度量，支持'euclidean'（欧几里得距离）、
                                'manhattan'（曼哈顿距离）和'chebyshev'（切比雪夫距离）。
        """
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k must be a positive integer.")
        if distance_metric not in ['euclidean', 'manhattan', 'chebyshev']:
            raise ValueError("Unsupported distance metric. Use 'euclidean', 'manhattan', or 'chebyshev'.")
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        训练KNN模型。由于KNN是懒惰学习算法，训练阶段实际上只是存储训练数据。

        :param X_train: 训练数据的特征。
        :param y_train: 训练数据的标签。
        """
        if X_train.ndim != 2 or y_train.ndim != 1:
            raise ValueError("X_train must be a 2D array and y_train must be a 1D array.")
        if len(X_train) != len(y_train):
            raise ValueError("The number of samples and labels must match.")
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        """
        计算两个样本之间的欧几里得距离。

        :param x1: 第一个样本的特征向量。
        :param x2: 第二个样本的特征向量。
        :return: 两个样本之间的欧几里得距离。
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        """
        计算两个样本之间的曼哈顿距离。

        :param x1: 第一个样本的特征向量。
        :param x2: 第二个样本的特征向量。
        :return: 两个样本之间的曼哈顿距离。
        """
        return np.sum(np.abs(x1 - x2))

    def _chebyshev_distance(self, x1, x2):
        """
        计算两个样本之间的切比雪夫距离。

        :param x1: 第一个样本的特征向量。
        :param x2: 第二个样本的特征向量。
        :return: 两个样本之间的切比雪夫距离。
        """
        return np.max(np.abs(x1 - x2))

    def _compute_distance(self, x1, x2):
        """
        根据指定的距离度量计算两个样本之间的距离。

        :param x1: 第一个样本的特征向量。
        :param x2: 第二个样本的特征向量。
        :return: 两个样本之间的距离。
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'chebyshev':
            return self._chebyshev_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric.")

    def predict(self, X_test):
        """
        对测试数据进行预测。

        :param X_test: 测试数据的特征。
        :return: 预测的标签数组。
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")
        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array.")
        predicted_labels = [self._predict(x) for x in X_test]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        对单个测试样本进行预测。

        :param x: 单个测试样本的特征向量。
        :return: 预测的标签。
        """
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def save(self, filename):
        """
        将KNN模型保存到文件。

        :param filename: 保存模型的文件名。
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        从文件加载KNN模型。

        :param filename: 从哪个文件加载模型。
        :return: 加载的KNN模型。
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)


class ModelEvaluator:
    def __init__(self, model):
        """
        初始化模型评估器，用于计算模型的各种评估指标。

        :param model: 被评估的模型。
        """
        self.model = model

    def accuracy_score(self, y_true, y_pred):
        """
        计算准确率，即正确预测的样本数占总样本数的比例。

        :param y_true: 真实的标签数组。
        :param y_pred: 预测的标签数组。
        :return: 准确率。
        """
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    def recall_score(self, y_true, y_pred):
        """
        计算召回率，即正确识别为正类的样本数占所有实际正类样本数的比例。

        :param y_true: 真实的标签数组。
        :param y_pred: 预测的标签数组。
        :return: 召回率。
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0

    def precision_score(self, y_true, y_pred):
        """
        计算精确率，即正确识别为正类的样本数占所有预测为正类样本数的比例。

        :param y_true: 真实的标签数组。
        :param y_pred: 预测的标签数组。
        :return: 精确率。
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0

    def f1_score(self, y_true, y_pred):
        """
        计算F1分数，它是精确率和召回率的调和平均数。

        :param y_true: 真实的标签数组。
        :param y_pred: 预测的标签数组。
        :return: F1分数。
        """
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def k_fold_cross_validation(self, X, y, k=5):
        """
        进行k折交叉验证，将数据集分为k个子集，每次留出一个子集作为测试集，其余作为训练集。

        :param X: 特征数据。
        :param y: 标签数据。
        :param k: 折数。
        :return: 包含每一折的评估结果的列表。
        """
        fold_size = len(y) // k
        results = []

        for fold in range(k):
            start = fold * fold_size
            end = start + fold_size if fold != k - 1 else len(y)
            X_train = np.concatenate([X[:start], X[end:]])
            y_train = np.concatenate([y[:start], y[end:]])
            X_test = X[start:end]
            y_test = y[start:end]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            results.append({
                'accuracy': self.accuracy_score(y_test, y_pred),
                'recall': self.recall_score(y_test, y_pred),
                'precision': self.precision_score(y_test, y_pred),
                'f1': self.f1_score(y_test, y_pred)
            })

        return results
