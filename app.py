from flask import Flask, request, jsonify, render_template, send_from_directory  # Flask相关模块导入
import os  # 操作系统模块
import threading  # 线程模块，用于并发处理
from SOTT_NN import NN_us  # 导入自定义的神经网络类
import pandas as pd  # 数据处理模块Pandas
import glob  # 文件模式匹配模块
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置环境变量，解决多重库加载问题

app = Flask(__name__)  # 初始化Flask应用

model = NN_us()  # 创建神经网络模型实例

@app.route('/')
def index():
    """处理主页请求，返回index.html页面"""
    return render_template('index.html')

@app.route('/compile', methods=['POST'])
def compile_model():
    """
    接收前端发来的编译请求，解析请求数据并编译模型。
    数据包括优化器、学习率、损失函数、学习方式等参数。
    """
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

    # 使用解析的数据编译模型
    model.compile(optimizer=optimizer, learning_rate=learning_rate, loss=loss, lr_schedule=learning_style, steps=steps, init_method=init_method)
    return jsonify({'status': 'Model compiled'})

@app.route('/add_conv', methods=['POST'])
def add_conv():
    """
    接收前端请求，向模型中添加卷积层。
    数据包括输入通道数、输出通道数、卷积核大小、步幅、填充方式等参数。
    """
    data = request.json
    print(data)

    input_channels = data['input_channels']
    output_channels = data['output_channels']
    kernel_size = data['kernel_size']
    stride = data.get('stride', 1)
    padding = data.get('padding', 0)
    conv = False  # 是否使用卷积核
    model.add_conv(input_channels, output_channels, kernel_size, stride, padding, conv)
    return jsonify({'status': 'Conv layer added'})

@app.route('/add_max_pool', methods=['POST'])
def add_max_pool():
    """
    接收前端请求，向模型中添加最大池化层。
    数据包括池化大小和步幅。
    """
    data = request.json
    pool_size = data['pool_size']
    stride = data['stride']
    model.add_max_pool(pool_size, stride)
    return jsonify({'status': 'Max pool layer added'})

@app.route('/add_batch_norm', methods=['POST'])
def add_batch_norm():
    """
    接收前端请求，向模型中添加批量归一化层。
    数据包括特征数量。
    """
    data = request.json
    num_features = data['num_features']
    model.add_batch_norm(num_features)
    return jsonify({'status': 'Batch norm layer added'})

@app.route('/add_flatten', methods=['POST'])
def add_flatten():
    """接收前端请求，向模型中添加平坦化层。"""
    model.add_flatten()
    return jsonify({'status': 'Flatten layer added'})

@app.route('/add_layer', methods=['POST'])
def add_layer():
    """
    接收前端请求，向模型中添加全连接层。
    数据包括输入维度和输出维度。
    """
    data = request.json
    input_dim = data['input_dim']
    output_dim = data['output_dim']
    model.add_layer(input_dim, output_dim)
    return jsonify({'status': 'Layer added'})

@app.route('/add_activation', methods=['POST'])
def add_activation():
    """
    接收前端请求，向模型中添加激活层。
    数据包括激活函数类型。
    """
    data = request.json
    activation_str = data['activation']
    model.add_activation(activation_str)
    return jsonify({'status': 'Activation layer added'})

@app.route('/add_residual_block', methods=['POST'])
def add_residual_block():
    """
    接收前端请求，向模型中添加残差块。
    数据包括输入通道数、输出通道数、步幅等参数。
    """
    data = request.json
    input_channels = data['input_channels']
    output_channels = data['output_channels']
    stride = data.get('stride', 1)
    use_batch_norm = False  # 是否使用批量归一化
    model.add_residual_block(input_channels, output_channels, stride, use_batch_norm)
    return jsonify({'status': 'Residual block added'})

@app.route('/add_dropout', methods=['POST'])
def add_dropout():
    """
    接收前端请求，向模型中添加Dropout层。
    数据包括丢弃率。
    """
    data = request.json
    p = data['p']
    model.add_dropout(p)
    return jsonify({'status': 'Dropout layer added'})

@app.route('/fit', methods=['POST'])
def fit_model():
    """
    接收前端请求，开始模型训练。
    数据包括训练周期数、是否保存指标、保存目录等。
    """
    data = request.json
    epochs = data.get('epochs', 100)
    save_metrics = data.get('save_metrics', None)
    save_dir = data.get('save_dir', './training_plots')

    # 启动一个新线程进行训练，以防止阻塞主线程
    train_thread = threading.Thread(target=model.fit, kwargs={'epochs': epochs, 'save_metrics': save_metrics, 'save_dir': save_dir})
    train_thread.start()

    return jsonify({'status': 'Model training started'}), 202

@app.route('/get_layers_inf', methods=['GET'])
def get_layers_inf():
    """获取当前模型的层次信息并返回为JSON格式"""
    layers_inf = model.get_layers_inf()
    return jsonify({'layers': layers_inf})

CSV_FILE_PATH = "D:/Awork/myProjectSum/pythonProject/Stride_of_the_Titan_copy/Stride_of_the_Titan/training_metrics.csv"  # 设置CSV文件路径

@app.route('/latest_data', methods=['GET'])
def latest_data():
    """
    读取训练过程中生成的最新指标数据，并返回为JSON格式。
    """
    df = pd.read_csv(CSV_FILE_PATH)
    data = {
        "epoch": df["Epoch"].tolist(),
        "accuracy": df["Accuracy"].tolist(),
        "loss": df["Loss"].tolist()
    }
    return jsonify({"data": data})

if __name__ == '__main__':
    app.run(debug=True)  # 以调试模式启动Flask应用
