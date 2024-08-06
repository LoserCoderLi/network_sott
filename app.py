from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import threading
from SOTT_NN import NN_us
import pandas as pd
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


app = Flask(__name__)

model = NN_us()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compile', methods=['POST'])
def compile_model():
    data = request.json
    print(data)
    optimizer = data.get('optimizer', 'sgd')
    learning_rate = data.get('learning_rate', 0.01)
    loss = data.get('loss', 'categorical_crossentropy')
    learning_style = data.get('learning_style', 'normal')
    step_min = data.get('step_min', 1000)
    step_max = data.get('step_max', 10000)
    steps = [step_min, step_max]
    init_method = data.get('init_method', 'he')
    print(optimizer, learning_rate, loss, learning_style)
    print(type(optimizer), type(learning_rate), type(loss))

    model.compile(optimizer=optimizer, learning_rate=learning_rate, loss=loss, lr_schedule=learning_style, steps=steps, init_method=init_method)
    return jsonify({'status': 'Model compiled'})

@app.route('/add_conv', methods=['POST'])
def add_conv():
    data = request.json
    print(data)

    input_channels = data['input_channels']
    output_channels = data['output_channels']
    kernel_size = data['kernel_size']
    stride = data.get('stride', 1)
    padding = data.get('padding', 0)
    conv = False
    model.add_conv(input_channels, output_channels, kernel_size, stride, padding, conv)
    return jsonify({'status': 'Conv layer added'})

@app.route('/add_max_pool', methods=['POST'])
def add_max_pool():
    data = request.json
    pool_size = data['pool_size']
    stride = data['stride']
    model.add_max_pool(pool_size, stride)
    return jsonify({'status': 'Max pool layer added'})

@app.route('/add_batch_norm', methods=['POST'])
def add_batch_norm():
    data = request.json
    num_features = data['num_features']
    model.add_batch_norm(num_features)
    return jsonify({'status': 'Batch norm layer added'})

@app.route('/add_flatten', methods=['POST'])
def add_flatten():
    model.add_flatten()
    return jsonify({'status': 'Flatten layer added'})

@app.route('/add_layer', methods=['POST'])
def add_layer():
    data = request.json
    input_dim = data['input_dim']
    output_dim = data['output_dim']
    model.add_layer(input_dim, output_dim)
    return jsonify({'status': 'Layer added'})

@app.route('/add_activation', methods=['POST'])
def add_activation():
    data = request.json
    activation_str = data['activation']
    model.add_activation(activation_str)
    return jsonify({'status': 'Activation layer added'})

@app.route('/add_residual_block', methods=['POST'])
def add_residual_block():
    data = request.json
    input_channels = data['input_channels']
    output_channels = data['output_channels']
    stride = data.get('stride', 1)
    # use_batch_norm = data.get('use_batch_norm', False)
    use_batch_norm = False
    model.add_residual_block(input_channels, output_channels, stride, use_batch_norm)
    return jsonify({'status': 'Residual block added'})

@app.route('/add_dropout', methods=['POST'])
def add_dropout():
    data = request.json
    p = data['p']
    model.add_dropout(p)
    return jsonify({'status': 'Dropout layer added'})

@app.route('/fit', methods=['POST'])
def fit_model():
    data = request.json
    epochs = data.get('epochs', 100)
    save_metrics = data.get('save_metrics', None)
    save_dir = data.get('save_dir', './training_plots')

    # 启动一个新的线程进行模型训练
    train_thread = threading.Thread(target=model.fit, kwargs={'epochs': epochs, 'save_metrics': save_metrics, 'save_dir': save_dir})
    train_thread.start()

    return jsonify({'status': 'Model training started'}), 202

@app.route('/get_layers_inf', methods=['GET'])
def get_layers_inf():
    layers_inf = model.get_layers_inf()
    return jsonify({'layers': layers_inf})

CSV_FILE_PATH = "D:/Awork/myProjectSum/pythonProject/Stride_of_the_Titan_copy/Stride_of_the_Titan/training_metrics.csv"  # 设置你的CSV文件路径

@app.route('/latest_data', methods=['GET'])
def latest_data():
    df = pd.read_csv(CSV_FILE_PATH)
    data = {
        "epoch": df["Epoch"].tolist(),
        "accuracy": df["Accuracy"].tolist(),
        "loss": df["Loss"].tolist()
    }
    return jsonify({"data": data})



if __name__ == '__main__':
    app.run(debug=True)
