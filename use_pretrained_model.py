import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer
import requests
import os
from myqwen import Qwen25VLModel

# 确保有safetensors库
try:
    import safetensors
except ImportError:
    print("请先安装safetensors: pip install safetensors")
    exit(1)

def load_image(image_path_or_url: str) -> Image.Image:
    """加载图像，支持本地路径或URL"""
    if image_path_or_url.startswith(('http://', 'https://')):
        # 从URL加载
        response = requests.get(image_path_or_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
    else:
        # 从本地路径加载
        image = Image.open(image_path_or_url)
    
    # 转换为RGB模式（如果是RGBA或其他模式）
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def preprocess_image(image: Image.Image, target_size: int = 224) -> torch.Tensor:
    """将PIL图像预处理为模型输入张量"""
    # 调整图像大小
    if image.width != target_size or image.height != target_size:
        image = image.resize((target_size, target_size))
    
    # 转换为numpy数组，并进行归一化
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # 转换为[C,H,W]格式
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)  # [1,C,H,W]
    
    return img_tensor

def load_pretrained_model(model_path: str = "pretrained_myqwen.pth", device: str = "cpu") -> Qwen25VLModel:
    """
    加载预训练模型
    
    Args:
        model_path: 预训练权重路径
        device: 运行设备
        
    Returns:
        加载了权重的模型
    """
    # 加载配置
    config_path = "models/models--remyxai--SpaceQwen2.5-VL-3B-Instruct/snapshots/ff71e2f363d635f5dbbd655bf11ac7f976f94c7d/config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
    else:
        # 默认配置
        config = {
            "hidden_size": 2048,
            "num_hidden_layers": 36,
            "num_attention_heads": 16,
            "intermediate_size": 11008,
            "max_position_embeddings": 128000,
            "vocab_size": 151936,
            "vision_config": {
                "hidden_size": 1280
            }
        }
    
    # 初始化模型
    model = Qwen25VLModel(
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=config.get("vision_config", {}).get("hidden_size", 1280),
        vision_depth=32,  # ViT层数
        vision_num_heads=16,
        vision_mlp_ratio=4.0,
        window_size=8,
        llm_dim=config.get("hidden_size", 2048),
        llm_depth=config.get("num_hidden_layers", 36),
        llm_num_heads=config.get("num_attention_heads", 16),
        llm_mlp_ratio=float(config.get("intermediate_size", 11008) / config.get("hidden_size", 2048)),
        max_position_embeddings=config.get("max_position_embeddings", 128000),
        vocab_size=config.get("vocab_size", 151936),
    )
    
    # 加载预训练权重
    if os.path.exists(model_path):
        print(f"加载预训练权重: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"找不到预训练权重: {model_path}，使用随机初始化")
    
    # 移至指定设备
    model = model.to(device)
    model.eval()
    
    return model

def initialize_tokenizer(tokenizer_path: str = "models/models--remyxai--SpaceQwen2.5-VL-3B-Instruct/snapshots/ff71e2f363d635f5dbbd655bf11ac7f976f94c7d"):
    """初始化分词器"""
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    return tokenizer

def generate_for_image(
    model: Qwen25VLModel,
    tokenizer,
    image: torch.Tensor,
    prompt: str = "描述这张图片中的内容。",
    max_length: int = 100,
    device: str = "cpu"
) -> str:
    """
    为图像生成描述
    
    Args:
        model: 模型
        tokenizer: 分词器
        image: 图像张量 [1, 3, H, W]
        prompt: 提示文本
        max_length: 最大生成长度
        device: 运行设备
        
    Returns:
        生成的文本
    """
    # 编码提示文本
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 图像张量移至设备
    image = image.to(device)
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            vision_input=image,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 解码生成的token
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    model = load_pretrained_model(device=device)
    
    # 初始化分词器
    tokenizer = initialize_tokenizer()
    
    # 准备示例图像
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
    print(f"加载示例图像: {image_url}")
    
    image = load_image(image_url)
    image_tensor = preprocess_image(image)
    
    # 生成图像描述
    prompt = "描述这张图片中的内容。请详细介绍主体、背景和整体氛围。"
    print(f"\n提示: {prompt}")
    
    generated_text = generate_for_image(
        model=model,
        tokenizer=tokenizer,
        image=image_tensor,
        prompt=prompt,
        device=device
    )
    
    print("\n生成的描述:")
    print(generated_text)
    
    # 测试另一个示例
    image_url = "https://images.unsplash.com/photo-1640935249932-18c866040ba3"
    print(f"\n加载另一个示例图像: {image_url}")
    
    image = load_image(image_url)
    image_tensor = preprocess_image(image)
    
    prompt = "这张图片展示了什么？有哪些主要元素？"
    print(f"\n提示: {prompt}")
    
    generated_text = generate_for_image(
        model=model,
        tokenizer=tokenizer,
        image=image_tensor,
        prompt=prompt,
        device=device
    )
    
    print("\n生成的描述:")
    print(generated_text)


if __name__ == "__main__":
    main() 