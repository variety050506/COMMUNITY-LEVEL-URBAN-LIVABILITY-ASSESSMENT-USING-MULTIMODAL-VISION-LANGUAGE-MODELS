import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from peft import PeftModel

# 配置
MODEL_PATH = "./deepseek-7b"  # 基础模型路径
LORA_PATH = "./deepseek_finetuned2"  # 微调后的LoRA权重路径
TEMPLATE = """请根据以下POI数据生成分析报告：
{POI_DATA}
只输出以下8项指标，输出格式必须完全一致，不得添加其他内容：
1. 潜在环境风险: [低/中/高]  
2. 教育设施覆盖度: [失衡/合理/优秀]  
3. 医疗资源可达性: [受限/一般/便捷]  
4. 金融行政便利性: [低下/普通/优越]  
5. 商业同质化程度: [严重/中等/轻微]  
6. 社区治理能力: [薄弱/适中/强大]  
7. 宗教种类: [单一/多元]  
8. 历史命名密度: [稀疏/普通/密集]

最终输出格式应当为：
1. 潜在环境风险: 
2. 教育设施覆盖度: 
3. 医疗资源可达性:   
4. 金融行政便利性:  
5. 商业同质化程度:  
6. 社区治理能力: 
7. 宗教种类: 
8. 历史命名密度: 
"""

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    torch_dtype=torch.float16
)
model.eval()

# 创建pipeline时不指定device参数
generator = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16
)

def process_poi_data(poi_data):
    """处理POI数据并生成报告"""
    prompt = TEMPLATE.replace("{POI_DATA}", poi_data)
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    result = generator(
        formatted_prompt,
        max_new_tokens=4096,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    full_output = result[0]["generated_text"]
    assistant_output = full_output.split("<|assistant|>\n")[-1]
    return assistant_output

def process_files(input_dir="wuhan_data", output_dir="wuhan_result"):
    """处理输入文件夹中的所有文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 如果结果文件已存在且非空，跳过处理
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Skipped (already processed): {filename}")
                continue
            
            print(f"Processing file: {filename}")
            
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    poi_data = f.read()
                
                result = process_poi_data(poi_data)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result)
                
                print(f"Result saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_directory = "dp_shanghai"
    output_directory = "shanghai_result"
    
    print("Starting POI data processing...")
    process_files(input_directory, output_directory)
    print("All files processed!")
