import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import requests
from typing import List, Dict, Union, Optional, Tuple

# 导入自定义模型
from myqwen import Qwen25VLModel
from modules.visual_encoder import VisualEncoder
from modules.m_rope import MRoPE
from modules.video_porocessor import VideoProcessor

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
    img_array = torch.from_numpy(np.array(image)).float()
    img_array = img_array / 255.0  # 归一化到[0,1]
    
    # 转换为[C,H,W]格式
    img_tensor = img_array.permute(2, 0, 1)
    
    # 添加批次维度
    img_tensor = img_tensor.unsqueeze(0)  # [1,C,H,W]
    
    return img_tensor

def main():
    # 初始化模型
    print("正在初始化Qwen2.5-VL模型...")
    model = Qwen25VLModel(
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        vision_depth=24,
        vision_num_heads=16,
        vision_mlp_ratio=4.0,
        window_size=8,
        llm_dim=2048,  # 3B模型使用2048维度
        llm_depth=32,
        llm_num_heads=32,
        llm_mlp_ratio=4.0,
        max_position_embeddings=4096,
        vocab_size=151936,  # 与Qwen2.5词表大小一致
    )
    
    # 将模型设置为评估模式
    model.eval()
    
    # 演示图像处理
    print("\n图像处理演示:")
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
    print(f"正在下载示例图像: {image_url}")
    
    # 加载并预处理图像
    try:
        image = load_image(image_url)
        image_tensor = preprocess_image(image)
        print(f"图像处理完成, 形状: {image_tensor.shape}")
    except Exception as e:
        print(f"图像加载失败: {e}")
        return
    
    # 准备文本输入
    print("\n准备模型输入...")
    input_text = "请描述这张图片中的内容。"
    # 这里简化处理，实际上需要使用分词器处理文本
    dummy_input_ids = torch.ones((1, 10), dtype=torch.long)  # 假设的输入ID
    
    # 使用模型处理
    print("\n运行模型推理...")
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=dummy_input_ids,
                vision_input=image_tensor,
                is_video=False,
            )
            print("模型推理完成!")
            print(f"输出隐藏状态形状: {outputs['hidden_states'].shape}")
            print(f"输出logits形状: {outputs['logits'].shape}")
        except Exception as e:
            print(f"模型推理失败: {e}")
    
    print("\nQwen2.5-VL模型架构演示完成!")
    print("注意：这是一个未经训练的模型架构演示，实际使用需要加载预训练权重。")

if __name__ == "__main__":
    import numpy as np
    main() 