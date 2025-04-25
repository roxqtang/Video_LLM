# Use a pipeline as a high-level helper
from transformers import pipeline
import os

# 在执行此脚本之前，请先运行 set_cache.ps1 设置环境变量
# 或者通过系统设置永久性地设置HF_HOME和HF_HUB_CACHE环境变量

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("image-text-to-text", 
                model="remyxai/SpaceQwen2.5-VL-3B-Instruct")  # 无需指定cache_dir，将使用环境变量
#pipe(messages)
print(pipe.model)

# 打印当前的环境变量设置，以便确认使用了正确的缓存目录
print("\n当前的Hugging Face缓存设置:")
print(f"HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', '未设置')}") 