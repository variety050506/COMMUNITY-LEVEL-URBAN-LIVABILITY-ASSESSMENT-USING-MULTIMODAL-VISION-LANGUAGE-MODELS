import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

# 允许超大影像
Image.MAX_IMAGE_PIXELS = None

# =========================
# 1. 模型加载
# =========================
model_path = "Janus-Pro-7B"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

# =========================
# 2. 路径设置
# =========================
image_folder = ""
output_folder = ""
poi_csv_path = ""

os.makedirs(output_folder, exist_ok=True)

# =========================
# 3. 固定提示词（你原始设计）
# =========================
base_prompt = (
    "请结合以上提供的街景影像信息以及该地块的POI功能构成，对该地块的人居环境进行综合评估。"
    "评估需从以下方面进行分析："
)

# =========================
# 4. 读取 POI CSV
# =========================
poi_df = pd.read_csv(poi_csv_path)
poi_df["block_id"] = poi_df["block_id"].astype(int)
poi_df = poi_df.set_index("block_id")
poi_columns = poi_df.columns.tolist()

def build_poi_description(block_id: int) -> str:
    """把某地块的 POI 数量转成一句自然语言"""
    if block_id not in poi_df.index:
        return "该地块未统计到POI信息。"

    row = poi_df.loc[block_id]
    parts = [
        f"{col}{int(row[col])}个"
        for col in poi_columns
        if row[col] > 0
    ]

    if not parts:
        return "该地块内未统计到POI。"

    return "该地块内POI数量如下：" + "，".join(parts) + "。"

# =========================
# 5. 推理函数
# =========================
def process_image_with_prompt(image_path, poi_desc, prompt):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{poi_desc}\n{prompt}",
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

# =========================
# 6. 批量处理影像
# =========================
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(image_extensions):
        continue

    base_name = os.path.splitext(image_name)[0]

    try:
        block_id = int(base_name)   # ★ 1.tif → block_id = 1
    except ValueError:
        print(f"无法解析地块ID，跳过: {image_name}")
        continue

    image_path = os.path.join(image_folder, image_name)
    output_path = os.path.join(output_folder, f"{base_name}.txt")

    if os.path.exists(output_path):
        print(f"已存在，跳过: {base_name}.txt")
        continue

    poi_desc = build_poi_description(block_id)

    try:
        answer = process_image_with_prompt(image_path, poi_desc, base_prompt)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(answer)
        print(f"处理完成: {image_name}")
    except Exception as e:
        print(f"处理失败 {image_name}: {e}")

print("全部影像处理完成")
