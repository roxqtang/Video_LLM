#!/usr/bin/env python
# 将已下载的Hugging Face模型从默认缓存复制到本地项目目录
import os
import shutil
import time
import argparse
from pathlib import Path

def get_default_cache_dir():
    """获取默认的Hugging Face模型缓存目录"""
    cache_home = os.path.expanduser("~/.cache")
    if os.name == "nt":  # Windows
        cache_home = os.path.expanduser("~/.cache")
    
    if "XDG_CACHE_HOME" in os.environ:
        cache_home = os.environ["XDG_CACHE_HOME"]
    
    # 默认模型缓存位置
    return os.path.join(cache_home, "huggingface", "hub")

def copy_model_files(model_id, source_dir, target_dir, verbose=True):
    """
    将模型文件从源目录复制到目标目录
    
    Args:
        model_id: 模型ID，例如 "meta-llama/Llama-2-7b-hf"
        source_dir: 源目录（默认缓存位置）
        target_dir: 目标目录（本地项目路径）
        verbose: 是否显示详细信息
    
    Returns:
        bool: 操作是否成功
    """
    # 在缓存中查找模型
    model_dirs = []
    model_id_path = model_id.replace("/", "--")
    
    # 检查模型缓存目录中的模型
    models_path = os.path.join(source_dir, "models--" + model_id_path)
    
    # 检查标准模型路径
    if os.path.exists(models_path):
        # 查找快照目录
        snapshots_dir = os.path.join(models_path, "snapshots")
        if os.path.exists(snapshots_dir):
            # 获取所有快照
            for snapshot in os.listdir(snapshots_dir):
                model_dirs.append(os.path.join(snapshots_dir, snapshot))
        else:
            # 如果没有快照目录，使用模型主目录
            model_dirs.append(models_path)
    
    # 如果没有找到任何模型目录，进行更广泛的搜索
    if not model_dirs:
        for root, dirs, _ in os.walk(source_dir):
            for dir_name in dirs:
                if model_id_path in dir_name:
                    model_dirs.append(os.path.join(root, dir_name))
    
    if not model_dirs:
        print(f"错误: 未在 {source_dir} 中找到模型 {model_id}")
        print("模型可能尚未下载，或者模型ID格式不正确。")
        return False
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    
    # 开始复制
    start_time = time.time()
    total_size = 0
    file_count = 0
    
    for model_dir in model_dirs:
        target_model_dir = os.path.join(target_dir, os.path.basename(model_dir))
        
        # 检查目标目录是否已存在
        if os.path.exists(target_model_dir):
            choice = input(f"目标目录已存在: {target_model_dir}\n覆盖现有文件? (y/n): ")
            if choice.lower() != 'y':
                print(f"跳过 {target_model_dir}")
                continue
            shutil.rmtree(target_model_dir)
        
        if verbose:
            print(f"正在复制 {model_dir} 到 {target_model_dir}...")
        
        # 复制整个目录
        shutil.copytree(model_dir, target_model_dir)
        
        # 计算复制的文件大小
        for root, _, files in os.walk(target_model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 打印结果
    if file_count > 0:
        print(f"\n复制完成!")
        print(f"- 复制的文件数: {file_count}")
        print(f"- 总大小: {total_size / (1024**3):.2f} GB")
        print(f"- 用时: {duration:.2f} 秒")
        return True
    else:
        print("未复制任何文件。")
        return False

def main():
    parser = argparse.ArgumentParser(description="将Hugging Face模型从默认缓存复制到本地目录")
    parser.add_argument("--model", type=str, required=True,
                        help="要复制的模型ID，例如 'meta-llama/Llama-2-7b-hf'")
    parser.add_argument("--source", type=str, default=None, 
                        help="源目录（默认为系统缓存路径）")
    parser.add_argument("--target", type=str, default="./models", 
                        help="目标目录，默认为当前目录下的models文件夹")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="显示详细复制信息")
    
    args = parser.parse_args()
    
    # 如果未指定源目录，使用默认缓存位置
    if args.source is None:
        args.source = get_default_cache_dir()
    
    # 打印操作信息
    print(f"模型ID: {args.model}")
    print(f"源目录: {args.source}")
    print(f"目标目录: {args.target}")
    
    # 执行复制
    success = copy_model_files(args.model, args.source, args.target, args.verbose)
    
    if success:
        print("\n操作成功完成！")
        # 提示使用本地缓存的方法
        abs_path = os.path.abspath(args.target)
        print("你现在可以设置以下环境变量来使用本地模型缓存:")
        print(f"export HF_HOME={abs_path}")
        print(f"export TRANSFORMERS_CACHE={abs_path}")
        print(f"export HF_HUB_CACHE={abs_path}")
    else:
        print("\n操作失败。")

if __name__ == "__main__":
    main() 