import pickle
import os
from SOTT_NN import NN_us
from tqdm import tqdm  # 导入tqdm库以显示进度条
import sys

def save_model_by_layer(model, folder):
    """
    将模型的每一层参数逐一保存到指定文件夹中。

    参数:
        model (NN_us): 要保存的模型对象。
        folder (str): 保存模型层参数的目标文件夹路径。
    """
    layers = model.get_layers()
    # 确保保存目录存在，如果不存在，创建它
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 保存每一层的参数，使用 tqdm 显示保存进度
    for i, layer in tqdm(enumerate(layers), total=len(layers), desc="Saving layers",
                         miniters=1, mininterval=0.01, file=sys.stdout):
        layer_params = layer.get_parameters()
        filepath = os.path.join(folder, f"layer_{i}.pkl")
        with open(filepath, 'wb') as file:
            pickle.dump(layer_params, file)

    # 保存编译参数到文件
    compile_params = {
        'optimizer': model.optimizer_name,
        'learning_rate': model.learning_rate,
        'loss': model.loss_name
    }
    compile_path = os.path.join(folder, "compile_params.pkl")
    with open(compile_path, 'wb') as file:
        pickle.dump(compile_params, file)
    print("Model compilation parameters saved.")

def load_model_from_folder(folder):
    """
    从指定文件夹中加载模型及其层参数。

    参数:
        folder (str): 包含模型层参数和编译参数的文件夹路径。

    返回:
        NN_us: 加载完成的模型对象。
    """
    model = NN_us()
    # 加载编译参数
    compile_path = os.path.join(folder, "compile_params.pkl")
    with open(compile_path, 'rb') as file:
        compile_params = pickle.load(file)
    model.compile(optimizer=compile_params['optimizer'],
                  learning_rate=compile_params['learning_rate'],
                  loss=compile_params['loss'])

    # 获取所有层文件，按层编号排序
    layer_files = sorted([f for f in os.listdir(folder) if f.startswith("layer_")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))

    # 使用 tqdm 显示加载进度
    for filename in tqdm(layer_files, desc="Loading layers", unit="layer"):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'rb') as file:
            layer_data = pickle.load(file)
            layer_type = layer_data['type']
            config = layer_data.get('config', {})

            # 根据层类型添加对应的层到模型中
            if layer_type == 'ConvolutionLayer':
                model.add_conv(**config)
            elif layer_type == 'Activation':
                model.add_activation(config['activation_type'])
            elif layer_type == 'MaxPoolingLayer':
                model.add_max_pool(**config)
            elif layer_type == 'BatchNormalization':
                model.add_batch_norm(config['num_features'])
            elif layer_type == 'FlattenLayer':
                model.add_flatten()
            elif layer_type == 'FullyConnectedLayer':
                model.add_layer(config['input_dim'], config['output_dim'])
            elif layer_type == 'Dropout':
                model.add_dropout(config['p'])
            elif layer_type == 'ResidualBlock':
                model.add_residual_block(input_channels=config['input_channels'],
                                         output_channels=config['output_channels'],
                                         stride=config.get('stride', 1),
                                         use_batch_norm=config.get('use_batch_norm', False))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # 如果模型的最后一层有 set_parameters 方法，则设置其参数
            if hasattr(model.layers[-1], 'set_parameters'):
                model.layers[-1].set_parameters(layer_data)

    return model
