import numpy as np
import pickle


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.01):
        if learning_rate <= 0 or num_iterations <= 0:
            raise ValueError("Learning rate and number of iterations must be greater than zero.")
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        z_safe = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_safe))

    def _initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01  # Small random numbers
        self.bias = 0

    def _compute_loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        m = y.shape[0]
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
        # Add L2 regularization
        loss += self.regularization_strength * 0.5 * np.sum(np.square(self.weights))
        return loss

    def fit(self, X, y):
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
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        z = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(z)
        return np.where(y_hat > 0.5, 1, 0)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _best_split(self, X, y):
        best_feature, best_value, best_gain = None, None, -1
        n_features = X.shape[1]
        parent_entropy = self._calculate_entropy(y)
        n = len(y)

        # Initialize feature importances array if not already done
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

            # Update feature importances
            if best_feature is not None:
                self.feature_importances_[best_feature] += gain

        return best_feature, best_value, best_gain

    def _build_tree(self, X, y, depth=0):
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
        """训练模型并重置/初始化特征重要性评分"""
        self.feature_importances_ = np.zeros(X.shape[1])  # Reset feature importances
        self.tree = self._build_tree(X, y, 0)
        self.feature_importances_ /= np.sum(self.feature_importances_)  # Normalize feature importances

    def predict(self, X):
        results = [self._predict_one(x, self.tree) for x in X]
        return np.array(results)

    def _predict_one(self, x, node):
        while node.get('feature') is not None:
            if x[node['feature']] <= node['value']:
                node = node['left']
            else:
                node = node['right']
        return node['label']

    def get_feature_importances(self):
        """返回特征重要性评分"""
        if self.feature_importances_ is None:
            raise ValueError("The model has not been fitted yet.")
        return self.feature_importances_

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k must be a positive integer.")
        if distance_metric not in ['euclidean', 'manhattan', 'chebyshev']:
            raise ValueError("Unsupported distance metric. Use 'euclidean', 'manhattan', or 'chebyshev'.")
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        if X_train.ndim != 2 or y_train.ndim != 1:
            raise ValueError("X_train must be a 2D array and y_train must be a 1D array.")
        if len(X_train) != len(y_train):
            raise ValueError("The number of samples and labels must match.")
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def _chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2))

    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'chebyshev':
            return self._chebyshev_distance(x1, x2)
        else:
            raise ValueError("Unsupported distance metric.")

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet.")
        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array.")
        predicted_labels = [self._predict(x) for x in X_test]
        return np.array(predicted_labels)

    def _predict(self, x):
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def accuracy_score(self, y_true, y_pred):
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

    def recall_score(self, y_true, y_pred):
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives != 0 else 0

    def precision_score(self, y_true, y_pred):
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives != 0 else 0

    def f1_score(self, y_true, y_pred):
        precision = self.precision_score(y_true, y_pred)
        recall = self.recall_score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def k_fold_cross_validation(self, X, y, k=5):
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

