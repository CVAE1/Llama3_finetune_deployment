import requests
import json


# Flask 服务器的预测端点的 URL
server_url = 'http://localhost:4892/predict'
#server_url = 'http://localhost:4891/predict'

# 定义 Prompt 模版
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

# 发送给服务器的 Prompt
prompt = prompt.format(
        "只用中文回答问题", # instruction
        "火烧赤壁 曹操为何不拨打119求救？", # input
        "", # output
    )

# 准备要在 POST 请求中发送的数据
data = {
    'prompt': prompt
}

# 发送 POST 请求给服务器
response = requests.post(server_url, json=data)

# 检查请求是否成功
if response.status_code == 200:
    # 输出来自服务器的回复
    print("Server response:", response.json())
else:
    print("Failed to get response from server, status code:", response.status_code)
