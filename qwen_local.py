# Use a pipeline as a high-level helper
from transformers import pipeline
import os

# 在执行此脚本之前，请先运行 set_local_cache.ps1 设置环境变量
# 运行方式: 
# 1. 先运行: .\set_local_cache.ps1
# 2. 然后运行: python qwen_local.py

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("image-text-to-text", 
                model="remyxai/SpaceQwen2.5-VL-3B-Instruct")  # 不指定cache_dir，将使用环境变量
#pipe(messages)
print(pipe.model)

# 打印当前的环境变量设置，以便确认使用了正确的缓存目录
print("\n当前的缓存目录设置:")
print(f"HF_HOME: {os.environ.get('HF_HOME', '未设置 - 将使用默认路径')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', '未设置 - 将使用默认路径')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置 - 将使用默认路径')}") 