from unsloth import FastLanguageModel
from transformers import TextStreamer


# 加载微调后的模型
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# 定义 Prompt 模版
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

# 预处理 Prompt
inputs = tokenizer(
[
    alpaca_prompt.format(
        "请用中文回答", # instruction
        "海绵宝宝的书法是不是叫做海绵体？", # input
        "", # output
    )
], return_tensors = "pt").to("cuda")

# 模型输出 token
text_streamer = TextStreamer(tokenizer)
token_ids = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128).squeeze(0)

# token 解码成文本
# results = tokenizer.decode(token_ids)
# print(results)
