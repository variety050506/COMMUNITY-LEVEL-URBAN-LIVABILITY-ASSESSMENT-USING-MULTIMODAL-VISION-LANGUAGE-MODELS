# train.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datasets import load_dataset
import torch
import json

# 配置
MODEL_PATH = "deepseek-7b"  # 本地7B模型路径
DATASET_PATH = "deepseek_finetune2.json"
MAX_LENGTH = 4096  # 修改为4096

# 1. 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 量化配置（根据显存情况调整）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4-bit量化
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# 2. 准备数据集
def prepare_data():
    dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split["train"], train_test_split["test"]

train_dataset, eval_dataset = prepare_data()

# 3. Tokenizer处理
def tokenize_function(samples):
    texts = [f"{prompt}\n{output}" for prompt, output in zip(samples["prompt"], samples["output"])]
    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 4. LoRA配置
peft_config = LoraConfig(
    r=16,  # 增大rank以适应更大上下文
    lora_alpha=32,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 明确指定目标模块
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,  # 减少epochs
    per_device_train_batch_size=1,  # 减小batch size
    gradient_accumulation_steps=2,  # 增加梯度累积
    learning_rate=1e-5,  # 更低的学习率
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    report_to="tensorboard"
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval
)

print("开始训练...")
trainer.train()
print("训练完成!")

# 7. 保存模型
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")

# 合并并保存完整模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

merged_model = PeftModel.from_pretrained(base_model, "./lora_adapter").merge_and_unload()
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")