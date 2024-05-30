from flask import Flask, request, jsonify
from gpt4all import GPT4All


# 创建 Flask 实例，Flask类实现了一个 WSGI 应用，它接收包或者模块的名字作为参数，但一般都是传递__name__
app = Flask(__name__)

# 指定微调后模型存放路径
model_path = ""
# 模型权重文件名
model = 'model-unsloth.Q4_K_M.gguf'

# 用指定的模型路径初始化 GPT4All 实例
gpt_instance = GPT4All(model, model_path=model_path)

# 使用 app.route 装饰器会将 URL 和执行的视图函数的关系保存到 app.url_map 属性上。处理 URL 和视图函数的关系的程序就是路由，这里的视图函数就是 predict
@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取数据
    data = request.json
    prompt = data.get('prompt', '')
    print("prompt= ", prompt)
    # 使用 GPT4All 进行预测
    response = gpt_instance.generate(prompt=prompt)
    print("server response= ", response)
    # 将预测作为 JSON 响应返回
    return jsonify(response)

if __name__ == '__main__':
    # 执行 app.run 就可以启动服务了。默认Flask只监听本地 127.0.0.1 这个地址，端口为5000
    # 而我们转发端口是 4892，所以需要指定 host 和 port 参数，0.0.0.0 表示监听所有地址，这样就可以在本机访问了
    # app.run(debug=True, host='0.0.0.0', port=4891)
    app.run(debug=True, host='127.0.0.1', port=4892)
