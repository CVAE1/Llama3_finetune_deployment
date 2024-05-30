from flask import Flask, request, jsonify
from gpt4all import GPT4All

app = Flask(__name__)

# Specify the path to the model folder
model_path = 'C:/Users/L_Win10/AppData/Local/nomic.ai/GPT4All'
model = 'model-unsloth.Q4_K_M.gguf'

# Initialize the GPT4All instance with the specified model path
gpt_instance = GPT4All(model, model_path=model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    prompt = data.get('prompt', '')
    print("prompt= ", prompt)

    # Perform prediction using GPT4All
    response = gpt_instance.generate(prompt=prompt)
    print("server response= ", response)

    # Return the prediction as a JSON response
    return jsonify(response)

if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=4891)
    app.run(debug=True, host='127.0.0.1', port=4892)