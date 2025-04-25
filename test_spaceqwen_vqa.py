# 使用Qwen模型测试视觉问答
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import os
import torch
from PIL import Image
import time
from qwen_vl_utils import process_vision_info

# 1. 设置模型缓存路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
data_dir = os.path.join(current_dir, "data")

# 确保文件夹存在
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# 设置环境变量，强制使用本地缓存
os.environ["TRANSFORMERS_CACHE"] = models_dir
os.environ["HF_HOME"] = models_dir
os.environ["HF_HUB_CACHE"] = models_dir

# 禁用默认缓存路径的警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 2. 加载模型 (使用local_files_only=False允许首次下载)
print("正在加载Qwen视觉模型...")
first_time = not os.path.exists(os.path.join(models_dir, "models--remyxai--SpaceQwen2.5-VL-3B-Instruct"))
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "remyxai/SpaceQwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")

# 3. 加载数据集
print("正在加载CLEVR数据集...")
# 设置数据集缓存
os.environ["HF_DATASETS_CACHE"] = data_dir

dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train", cache_dir=data_dir)
print(f"数据集加载完成，共有 {len(dataset)} 个样本")

# 打印缓存路径信息
print("\n当前使用的缓存路径:")
print(f"模型缓存: {models_dir}")
print(f"数据集缓存: {data_dir}")
print(f"环境变量 HF_HOME: {os.environ.get('HF_HOME')}")
print(f"环境变量 TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"环境变量 HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")

# 4. 测试模型（使用少量样本）
num_samples = 5  # 设置测试样本数量
results = []

print(f"\n开始测试 {num_samples} 个样本...")
for i in range(min(num_samples, len(dataset))):
    sample = dataset[i]
    image = sample['image']
    question = sample['problem']
    ground_truth = sample['solution']
    
    assert isinstance(image, Image.Image), f"image type: {type(image)}"
    
    print(f"\n样本 {i+1}:")
    print(f"问题: {question}")
    print(f"参考答案: {ground_truth}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"模型回答: {output_text[0]}")
    
    # 计时
    start_time = time.time()
    
    end_time = time.time()
    
    print(f"推理用时: {end_time - start_time:.2f} 秒")
    
    # 保存结果
    results.append({
        "index": i,
        "question": question,
        "ground_truth": ground_truth,
        "model_response": output_text[0],
        "time": end_time - start_time
    })

# 5. 输出结果汇总
print("\n\n结果汇总:")
print("-" * 50)
total_time = sum(result["time"] for result in results)
print(f"测试样本数: {len(results)}")
print(f"总用时: {total_time:.2f} 秒")
print(f"平均每个样本用时: {total_time / len(results):.2f} 秒")
print("-" * 50)

for result in results:
    print(f"样本 {result['index']+1}:")
    print(f"问题: {result['question']}")
    print(f"参考答案: {result['ground_truth']}")
    print(f"模型回答: {result['model_response']}")
    print("-" * 30) 