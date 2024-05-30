from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


# unsloth 库支持的 4 位预量化模型，下载速度提高四倍，没有 OOMs 问题。更多模型详见 at https://huggingface.co/unsloth
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
]

# 定义 Prompt 模版
# Instruction 描述模型应该执行的任务
# Input 描述任务的可选上下文或输入，例如，当指令是“总结下面的文章”时，输入就是文章
# Response 生成的 Instruction 的回答
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

# 加载模型
max_seq_length = 2048 # 内部支持 RoPE Scaling，因此可以任意选择
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 根据 Prompt 模版加载微调数据集
EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
dataset = load_dataset('json', data_files='common_sense_qa.json', split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 设置训练参数，进行模型修补并添加快速 LoRA 权重
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # 支持任意值，但是 = 0 已优化
    bias = "none",    # 支持任意值， 但是 = "none" 已优化
    # “unsloth” 使用的 VRAM 减少 30%，适合大 2 倍的 batch size！
    use_gradient_checkpointing = "unsloth", # 在很长的上下文中为 True 或 “unsloth”
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # 支持秩稳定的 LoRA
    loftq_config = None, # 以及 LoftQ
)
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)

# 开始训练
trainer.train()

# 保存LoRA模型
model.save_pretrained("lora_model") # 本地保存
# model.push_to_hub("your_name/lora_model", token = "...") # 在线保存到 hugging face，需要 token

# 合并模型并量化成4位gguf保存
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
#model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",) # 合并模型，保存为16位 hf
#model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "") # 合并4位 gguf，上传到 hugging face(需要账号token)
