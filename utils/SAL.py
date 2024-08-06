import pickle
import os
from SOTT_NN import NN_us
from tqdm import tqdm  # 导入tqdm库
import sys

def save_model_by_layer(model, folder):
    layers = model.get_layers()
    # 确保保存目录存在，如果不存在，创建它
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 保存层参数，使用 tqdm 包裹循环以显示进度条
    for i, layer in tqdm(enumerate(layers), total=len(layers), desc="Saving layers", miniters=1, mininterval=0.01, file=sys.stdout):
        layer_params = layer.get_parameters()
        filepath = os.path.join(folder, f"layer_{i}.pkl")
        with open(filepath, 'wb') as file:
            pickle.dump(layer_params, file)


    # 保存编译参数
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
    model = NN_us()
    # 加载编译参数
    compile_path = os.path.join(folder, "compile_params.pkl")
    with open(compile_path, 'rb') as file:
        compile_params = pickle.load(file)
    model.compile(optimizer=compile_params['optimizer'],
                  learning_rate=compile_params['learning_rate'],
                  loss=compile_params['loss'])

    layer_files = sorted([f for f in os.listdir(folder) if f.startswith("layer_")],
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    # model.compile(optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # 使用 tqdm 包裹循环以显示进度条
    for filename in tqdm(layer_files, desc="Loading layers", unit="layer"):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'rb') as file:
            layer_data = pickle.load(file)
            layer_type = layer_data['type']
            config = layer_data.get('config', {})

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

            if hasattr(model.layers[-1], 'set_parameters'):
                model.layers[-1].set_parameters(layer_data)

    return model
