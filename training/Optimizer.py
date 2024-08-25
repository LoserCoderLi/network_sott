import numpy as np


class LearningRateScheduler:
    def __init__(self, initial_lr, schedule_type='normal', warmup_steps=0, total_steps=10000):
        """
        初始化学习率调度器。
        
        参数:
        - initial_lr: 初始学习率
        - schedule_type: 调度类型，默认为 'normal'
        - warmup_steps: 预热步骤的数量，默认为 0
        - total_steps: 总训练步骤数，默认为 10000
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def get_lr(self, current_step=None):
                """
        根据当前训练步骤获取当前的学习率。
        
        参数:
        - current_step: 当前训练步骤，如果未指定则使用类中的 current_step
        
        返回:
        - lr: 当前的学习率
        """
        if current_step is not None:
            self.current_step = current_step

        if self.schedule_type == 'normal':
            return self.initial_lr
        elif self.schedule_type == 'rate_warmup':
            if self.current_step < self.warmup_steps:
                return self.initial_lr * (self.current_step / self.warmup_steps)
            else:
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return 0.5 * self.initial_lr * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError("Unknown schedule type")

    def step(self):
        self.current_step += 1
        """
        更新当前的训练步骤。
        """


'''
选择合适的优化器对于训练神经网络和其他机器学习模型至关重要。不同的优化器有不同的特性和适用场景。以下是一些常见优化器的简要介绍及其适用情况：

### 1. SGD (随机梯度下降)

- **特点**: 最经典和基础的优化器，每次更新只使用一个训练样本的梯度。
- **适用场景**: 当数据集相对简单或规模较小时，SGD通常表现良好。但在复杂的数据集和深层网络中可能表现不佳，因为它不包含加速收敛或调整学习率的机制。

### 2. Adam (自适应矩估计)

- **特点**: 结合了RMSprop和Momentum优化器的优点，调整学习率并保持过去梯度的一定比例，使其更稳定。
- **适用场景**: 适用于大多数非凸优化问题，特别是大数据集和高维空间问题。Adam是训练深度神经网络的首选优化器之一。

### 3. RMSprop

- **特点**: 通过除以梯度的滑动平均的平方根来调整学习率，适用于非平稳目标。
- **适用场景**: 适用于处理非常不稳定的数据集，如递归神经网络（RNN）等。

### 4. Adagrad

- **特点**: 根据参数的更新频率自动调整学习率，对于每个参数有不同的学习率。
- **适用场景**: 在处理稀疏数据时表现出色，如大规模文本数据。

### 选择建议

- 对于大多数常规问题，**Adam**是一个很好的起点。
- 如果您处理的是更大的、更复杂的网络，特别是深度学习任务，那么**Adam**或**RMSprop**可能是更好的选择。
- 对于稀疏数据集，如某些文本数据或图像处理问题，**Adagrad**可以有效。
- 当模型相对简单或者想要更多控制学习过程时，可以考虑使用**SGD**。

在实践中，选择哪种优化器往往需要通过实验来确定，因为它还取决于具体问题的特性以及模型的结构。不同优化器可能在不同的数据集和模型架构上表现出不同的性能。
'''
class Optimizers:
    class SGD:
        def __init__(self, learning_rate=0.01, scheduler=None):
            self.learning_rate = learning_rate
            self.scheduler = scheduler

        def update(self, params, grads):
            if self.scheduler:
                self.learning_rate = self.scheduler.get_lr()
                self.scheduler.step()
            params -= self.learning_rate * grads
            return params

    class Adam:
        def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_grad_norm=10.0,
                     scheduler=None):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.max_grad_norm = max_grad_norm
            self.scheduler = scheduler
            self.m = {}
            self.v = {}
            self.t = 0

        def update(self, params, grads):
            if id(params) not in self.m:
                self.m[id(params)] = np.zeros_like(params)
                self.v[id(params)] = np.zeros_like(params)

            self.t += 1  # Increment update count

            # Learning rate scheduling
            if self.scheduler:
                self.learning_rate = self.scheduler.get_lr(self.t)

            # Clip gradients to prevent gradient explosion
            grad_norm = np.linalg.norm(grads)
            if grad_norm > self.max_grad_norm:
                grads = grads * (self.max_grad_norm / grad_norm)

            # Update biased first moment estimate and biased second raw moment estimate
            self.m[id(params)] = self.beta1 * self.m[id(params)] + (1 - self.beta1) * grads
            self.v[id(params)] = self.beta2 * self.v[id(params)] + (1 - self.beta2) * (grads ** 2)

            # Compute bias-corrected first moment estimate and second raw moment estimate
            m_hat = self.m[id(params)] / (1 - self.beta1 ** self.t)
            v_hat = self.v[id(params)] / (1 - self.beta2 ** self.t)

            # Update parameters
            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Optional debugging output
            if self.t % 100 == 0:
                print(
                    f"Step {self.t}: Grad norm={grad_norm:.4f}, Param update norm={np.linalg.norm(self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)):.4f}")

            return params

    class RMSprop:
        def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8, scheduler=None):
            self.learning_rate = learning_rate
            self.rho = rho
            self.epsilon = epsilon
            self.s = None
            self.scheduler = scheduler

        def update(self, params, grads):
            if self.scheduler:
                self.learning_rate = self.scheduler.get_lr()
                self.scheduler.step()

            if self.s is None:
                self.s = np.zeros_like(params)

            self.s = self.rho * self.s + (1 - self.rho) * np.square(grads)
            params -= self.learning_rate * grads / (np.sqrt(self.s) + self.epsilon)
            return params

    class Adagrad:
        def __init__(self, learning_rate=0.01, epsilon=1e-8, scheduler=None):
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.s = None
            self.scheduler = scheduler

        def update(self, params, grads):
            if self.scheduler:
                self.learning_rate = self.scheduler.get_lr()
                self.scheduler.step()

            if self.s is None:
                self.s = np.zeros_like(params)

            self.s += np.square(grads)
            params -= self.learning_rate * grads / (np.sqrt(self.s) + self.epsilon)
            return params
