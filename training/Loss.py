import numpy as np

class LossFunctions:
    '''
    在机器学习和深度学习中，选择合适的损失函数是至关重要的，因为它直接影响到模型如何从训练数据中学习。不同的损失函数适用于不同的任务和数据类型。以下是您定义的损失函数的使用指南：

    ### 1. BinaryCrossEntropy
    - **适用于**: 二分类问题。
    - **场景**: 当您的任务是预测一个输出属于两个类别中的哪一个时，例如判断邮件是否为垃圾邮件。
    - **特点**: 对于每个样本，模型输出一个介于0到1之间的概率值。

    ### 2. CategoricalCrossEntropy
    - **适用于**: 多分类问题。
    - **场景**: 当您的任务是将输入分类到多个类别中的一个时，例如图像分类或文本标签分类。
    - **特点**: 每个样本的模型输出是一个概率分布，表示属于每个类别的概率。

    ### 3. MeanSquaredError
    - **适用于**: 回归问题。
    - **场景**: 当您的任务是预测连续值时，如房价预测、温度预测等。
    - **特点**: 试图最小化预测值和实际值之间的平方差。

    ### 4. HuberLoss
    - **适用于**: 回归问题。
    - **场景**: 与MSE类似的回归任务，但更加鲁棒，对异常值不那么敏感。
    - **特点**: 是MSE和MAE的折中选择，通过一个阈值来平衡对异常值的敏感度。

    ### 5. LogCoshLoss
    - **适用于**: 回归问题。
    - **场景**: 回归任务，尤其是在对预测误差的大值不想过于敏感时。
    - **特点**: 试图最小化预测值和实际值之差的双曲余弦的对数，对于大的误差比MSE更加平滑。

    ### 总结
    - 对于**分类任务**（二分类或多分类），通常使用**BinaryCrossEntropy** 或 **CategoricalCrossEntropy**。
    - 对于**回归任务**，常用的是 **MeanSquaredError**，而 **HuberLoss** 和 **LogCoshLoss** 在对异常值的处理上更加灵活和鲁棒。

    选择适当的损失函数取决于您的特定需求、任务类型和数据的特性。在实际应用中，可能需要根据模型在验证集上的表现来调整和选择最合适的损失函数。
    '''

    class BinaryCrossEntropy:
                """
        二分类交叉熵损失函数。

        参数:
        - y_true: 真实标签，值为0或1。
        - y_pred: 预测概率，值在0到1之间。

        返回:
        - loss: 二分类交叉熵损失值。
        """
        Attribute = "分类"

        def loss(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1. - y_true) * np.log(1. - y_pred))

        def grad(self, y_true, y_pred):
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -(y_true / y_pred - (1. - y_true) / (1. - y_pred))

    class CategoricalCrossEntropy:
                """
        多分类交叉熵损失函数。

        参数:
        - y_true: 真实标签的独热编码形式。
        - y_pred: 预测概率分布。

        返回:
        - loss: 多分类交叉熵损失值。
        """
        Attribute = "分类"

        def loss(self, y_true, y_pred):
            epsilon = 1e-7
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -np.sum(y_true * np.log(y_pred))

        def grad(self, y_true, y_pred):
            epsilon = 1e-7
            y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
            return -y_true / y_pred

    class CrossEntropyLossWithSoftmax:
        def __init__(self):
            self.epsilon = 1e-7
            self.grad_cache = None

        def softmax(self, logits):
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        def loss(self, y_true, logits):
            # 应用Softmax
            y_pred = self.softmax(logits)
            # 避免数值稳定性问题
            y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            # 计算交叉熵损失
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
            # 缓存前向传播的Softmax输出以用于反向传播
            self.grad_cache = y_pred
            return loss

        def grad(self, y_true, _):
            # 使用缓存的Softmax输出计算简化的梯度
            if self.grad_cache is None:
                raise ValueError("Must call loss before calling grad.")
            return self.grad_cache - y_true

    class MeanSquaredError:
                        """
        均方误差损失函数，常用于回归问题。

        参数:
        - y_true: 真实值。
        - y_pred: 预测值。

        返回:
        - loss: 均方误差值。
        """
        Attribute = "回归"

        def loss(self, y_true, y_pred):
            return np.mean(np.square(y_true - y_pred))

        def grad(self, y_true, y_pred):
            # print(y_pred.shape)
            y_true = y_true.reshape(-1, 1)
            # print(y_true.shape)
            return 2 * (y_pred - y_true) / y_true.shape[0]

    class HuberLoss:
               """
        Huber 损失函数，对异常值不敏感的回归损失。

        参数:
        - y_true: 真实值。
        - y_pred: 预测值。
        - delta: 阈值，用于调整损失函数的平滑程度。

        返回:
        - loss: Huber 损失值。
        """
        Attribute = "回归"

        def __init__(self, delta=1.0):
            self.delta = delta

        def loss(self, y_true, y_pred):
            error = y_true - y_pred
            is_small_error = np.abs(error) < self.delta
            squared_loss = np.square(error) / 2
            linear_loss = self.delta * np.abs(error) - self.delta ** 2 / 2
            return np.where(is_small_error, squared_loss, linear_loss)

        def grad(self, y_true, y_pred):
            error = y_true - y_pred
            is_small_error = np.abs(error) < self.delta
            squared_grad = error
            linear_grad = self.delta * np.sign(error)
            return np.where(is_small_error, squared_grad, linear_grad)

    class LogCoshLoss:
                """
        Log-cosh 损失函数，一种对大误差平滑处理的回归损失。

        参数:
        - y_true: 真实值。
        - y_pred: 预测值。

        返回:
        - loss: Log-cosh 损失值。
        """
        Attribute = "回归"

        def loss(self, y_true, y_pred):
            error = y_pred - y_true
            return np.mean(np.log(np.cosh(error + 1e-12)))

        def grad(self, y_true, y_pred):
            error = y_pred - y_true
            return np.tanh(error)
