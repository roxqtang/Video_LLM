# 评估Qwen模型在CLEVR数据集上的性能
from transformers import pipeline
from datasets import load_dataset
import os
import json
import re
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# 1. 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "models")
data_dir = os.path.join(current_dir, "data")
output_dir = os.path.join(current_dir, "evaluation_results")

# 确保文件夹存在
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 2. 辅助函数：从答案文本中提取关键信息
def extract_answer(text):
    """从模型回答中提取答案"""
    # 尝试提取<answer>标签中的内容
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 直接返回整个文本
    return text.strip()

# 3. 加载模型
print("正在加载Qwen视觉模型...")
pipe = pipeline("image-text-to-text", 
                model="remyxai/SpaceQwen2.5-VL-3B-Instruct", 
                cache_dir=models_dir)

# 4. 加载数据集
print("正在加载CLEVR数据集...")
dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train", cache_dir=data_dir)
print(f"数据集加载完成，共有 {len(dataset)} 个样本")

# 5. 评估参数
num_eval_samples = 10  # 评估样本数量
random_seed = 42       # 随机种子，确保可重复性

# 6. 采样评估数据
np.random.seed(random_seed)
eval_indices = np.random.choice(len(dataset), num_eval_samples, replace=False)
print(f"已随机选择 {num_eval_samples} 个样本进行评估")

# 7. 执行评估
results = []
correct_count = 0
total_time = 0

print("\n开始评估...")
for i, idx in enumerate(tqdm(eval_indices)):
    sample = dataset[idx]
    image = sample['image']
    question = sample['problem']
    ground_truth = sample['solution']
    extracted_ground_truth = extract_answer(ground_truth)
    
    # 构建提示
    messages = [
        {"role": "user", "content": f"Look at the image and answer the following question with only the answer in the format <answer>your answer</answer>: {question}"}
    ]
    
    # 模型推理
    start_time = time.time()
    response = pipe(messages, image)
    end_time = time.time()
    inference_time = end_time - start_time
    total_time += inference_time
    
    # 提取模型回答
    extracted_response = extract_answer(response)
    
    # 判断答案是否正确（简单字符串匹配）
    is_correct = (extracted_ground_truth.lower() == extracted_response.lower())
    if is_correct:
        correct_count += 1
    
    # 保存结果
    results.append({
        "sample_idx": idx,
        "question": question,
        "ground_truth": ground_truth,
        "extracted_ground_truth": extracted_ground_truth,
        "model_response": response,
        "extracted_response": extracted_response,
        "is_correct": is_correct,
        "inference_time": inference_time
    })
    
    # 打印当前样本的评估结果
    print(f"\n样本 {i+1}/{num_eval_samples} (数据集索引: {idx}):")
    print(f"问题: {question}")
    print(f"参考答案: {extracted_ground_truth}")
    print(f"模型回答: {extracted_response}")
    print(f"是否正确: {'✓' if is_correct else '✗'}")
    print(f"推理用时: {inference_time:.2f}秒")

# 8. 计算指标
accuracy = correct_count / num_eval_samples
avg_time = total_time / num_eval_samples

# 9. 输出评估结果
print("\n" + "="*50)
print("评估结果摘要:")
print("="*50)
print(f"评估样本数: {num_eval_samples}")
print(f"准确率: {accuracy:.2%}")
print(f"平均推理时间: {avg_time:.2f}秒/样本")
print("="*50)

# 10. 保存评估结果
timestamp = time.strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(output_dir, f"qwen_clevr_eval_{timestamp}.json")

with open(result_file, "w", encoding="utf-8") as f:
    json.dump({
        "metadata": {
            "model": "remyxai/SpaceQwen2.5-VL-3B-Instruct",
            "dataset": "MMInstruction/Clevr_CoGenT_ValB",
            "num_samples": num_eval_samples,
            "random_seed": random_seed,
            "timestamp": timestamp
        },
        "metrics": {
            "accuracy": accuracy,
            "avg_inference_time": avg_time
        },
        "samples": results
    }, f, ensure_ascii=False, indent=2)

print(f"评估结果已保存到: {result_file}") 