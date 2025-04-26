import argparse
import os
import json
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
import re

def evaluate_response(model_response, sample, dataset_name, threshold=2, batch_idx=0):
    """评估模型回答并返回结果信息"""
    result = {
        "index": batch_idx,
        "question": sample["problem"],
        "model_response": model_response,
        "ground_truth": sample["solution"]
    }
    
    if dataset_name == "CLEVR":
        ground_truth = sample["solution"]
        
        # 首先确定问题类型
        if is_boolean_answer(ground_truth):
            result["question_type"] = "boolean"
            # 尝试从响应中提取答案
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip().lower()
                result["pred_answer"] = extracted_answer
                is_correct = compare_answers(extracted_answer, ground_truth)
                result["correct"] = is_correct
                
                if is_correct:
                    print(f"答案评估: 布尔问题回答正确")
                    return result, "boolean", True
                else:
                    print(f"答案评估: 布尔问题回答错误 (模型: {extracted_answer}, 正确: {ground_truth})")
                    return result, "boolean", False
            else:
                print("答案评估: 无效答案格式 (未包含<answer>标签)")
                result["question_type"] = "invalid"
                return result, "invalid", None
                
        # 检查是否为数值答案
        elif extract_number_from_answer(ground_truth) is not None:
            result["question_type"] = "counting"
            # 尝试从响应中提取答案
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                try:
                    pred_number = extract_number_from_answer(extracted_answer)
                    true_number = extract_number_from_answer(ground_truth)
                    
                    if pred_number is not None and true_number is not None:
                        diff = abs(pred_number - true_number)
                        result["diff"] = diff
                        # 对CLEVR数据集，仅当完全正确时才算正确
                        is_correct = (diff == 0)
                        result["correct"] = is_correct
                        
                        if is_correct:
                            print(f"答案评估: 计数问题回答正确 ({pred_number})")
                            return result, "counting", True
                        else:
                            print(f"答案评估: 计数问题回答错误 (模型: {pred_number}, 正确: {true_number}, 差值: {diff})")
                            return result, "counting", False
                    else:
                        print("答案评估: 无法从回答中提取有效数字")
                        result["question_type"] = "invalid"
                        return result, "invalid", None
                except:
                    print("答案评估: 数值处理异常")
                    result["question_type"] = "invalid"
                    return result, "invalid", None
            else:
                print("答案评估: 无效答案格式 (未包含<answer>标签)")
                result["question_type"] = "invalid"
                return result, "invalid", None
                
        # 其他类型的问题
        else:
            result["question_type"] = "other"
            # 尝试从响应中提取答案
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                is_correct = compare_answers(extracted_answer, ground_truth)
                result["correct"] = is_correct
                
                if is_correct:
                    print("答案评估: 其他问题回答正确")
                    return result, "other", True
                else:
                    print(f"答案评估: 其他问题回答错误 (模型: {extracted_answer}, 正确: {ground_truth})")
                    return result, "other", False
            else:
                print("答案评估: 无效答案格式 (未包含<answer>标签)")
                result["question_type"] = "invalid"
                return result, "invalid", None
    
    elif dataset_name == "SAT":
        # 对于SAT数据集，检查是否为计数题
        if "is_counting" in sample and sample["is_counting"]:
            result["question_type"] = "counting"
            
            # 尝试从响应中提取答案
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                try:
                    pred_number = extract_number_from_answer(extracted_answer)
                    true_number = sample.get("number_answer")
                    
                    if pred_number is not None and true_number is not None:
                        diff = abs(pred_number - true_number)
                        result["diff"] = diff
                        # 对SAT数据集，使用阈值确定是否正确
                        # 注意：只有差值为0时才算完全正确，但差值小于等于阈值时视为在可接受范围内
                        is_completely_correct = (diff == 0)
                        is_within_threshold = (diff <= threshold)
                        result["completely_correct"] = is_completely_correct
                        result["correct"] = is_within_threshold
                        
                        if is_completely_correct:
                            print(f"答案评估: 计数问题完全正确 ({pred_number})")
                            return result, "counting", True
                        elif is_within_threshold:
                            print(f"答案评估: 计数问题在误差范围内 (模型: {pred_number}, 正确: {true_number}, 差值: {diff})")
                            return result, "counting", True
                        else:
                            print(f"答案评估: 计数问题超出误差范围 (模型: {pred_number}, 正确: {true_number}, 差值: {diff})")
                            return result, "counting", False
                    else:
                        print("答案评估: 无法从回答中提取有效数字")
                        result["question_type"] = "invalid"
                        return result, "invalid", None
                except Exception as e:
                    print(f"答案评估: 数值处理异常 - {e}")
                    result["question_type"] = "invalid"
                    return result, "invalid", None
            else:
                print("答案评估: 无效答案格式 (未包含<answer>标签)")
                result["question_type"] = "invalid"
                return result, "invalid", None
        else:
            # 非计数题，正常处理为"other"类型问题
            result["question_type"] = "other"
            # 尝试从响应中提取答案
            match = re.search(r'<answer>(.*?)</answer>', model_response, re.IGNORECASE | re.DOTALL)
            if match:
                extracted_answer = match.group(1).strip()
                result["pred_answer"] = extracted_answer
                
                # 非计数题，我们只需记录结果，不会影响计数题统计
                print("答案评估: 非计数题")
                return result, "other", None
            else:
                print("答案评估: 无效答案格式 (未包含<answer>标签)")
                result["question_type"] = "invalid"
                return result, "invalid", None
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def process_vision_info(message):
    """处理消息中的视觉信息（图像和视频）"""
    image_inputs = []
    video_inputs = None
    for item in message[0]["content"]:
        # 处理图像
        if item["type"] == "image":
            image_inputs.append(item["image"])
        # 处理视频
        elif item["type"] == "video":
            video_inputs = item["video"]
    
    # 统一输出格式
    if len(image_inputs) == 0:
        image_inputs = None
    elif len(image_inputs) == 1:
        image_inputs = image_inputs[0]
    
    return image_inputs, video_inputs

def process_sat_sample(sample):
    """处理SAT数据集样本（Hugging Face格式）"""
    try:
        question = sample['problem']
        solution = f"<answer>{sample['solution']}</answer>"
        ground_truth = sample.get('answer', sample.get('solution', ""))
        image = sample.get('image', None)
        
        # 对于没有image的样本，尝试从image_path加载
        if image is None and 'image_path' in sample:
            from PIL import Image
            try:
                image = Image.open(sample['image_path']).convert("RGB")
            except Exception as e:
                print(f"无法加载图像 {sample['image_path']}: {e}")
                return None
        
        # 检查是否为计数题（通过solution判断是否包含数字）
        is_counting = False
        number_in_solution = extract_number_from_answer(ground_truth)
        if number_in_solution is not None:
            is_counting = True
            
        # 返回处理后的样本
        return {
            'problem': question,
            'solution': solution,
            'image': image,
            'is_counting': is_counting,  # 新增字段，标记是否为计数题
            'number_answer': number_in_solution  # 新增字段，存储计数答案
        }
    except Exception as e:
        print(f"处理SAT样本时出错: {e}")
        return None

def extract_number_from_answer(ans):
    """从答案中提取数字"""
    # 尝试找到有明确标识的数字
    if ans is None:
        return None
    
    # 从文本中提取数字，支持多种不同的格式
    number_patterns = [
        r"<answer>(\d+(\.\d+)?)</answer>",  # <answer>5</answer>
        r"the answer is (\d+(\.\d+)?)",  # the answer is 5
        r"answer: (\d+(\.\d+)?)",  # answer: 5
        r"(\d+(\.\d+)?)",  # 任何数字
    ]
    
    for pattern in number_patterns:
        match = re.search(pattern, ans, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # 如果没有找到数字，检查数字的英文表示
    word_to_number = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    
    for word, number in word_to_number.items():
        if re.search(r'\b' + word + r'\b', ans, re.IGNORECASE):
            return float(number)
    
    return None

def is_boolean_answer(ans):
    """判断是否是布尔答案"""
    # 检查是否包含明确的布尔答案模式
    bool_patterns = [
        r"<answer>(yes|no|true|false)</answer>",
        r"\b(yes|no|true|false)\b",
    ]
    
    for pattern in bool_patterns:
        if re.search(pattern, ans.lower()):
            return True
    
    return False

def compare_answers(pred_answer, ground_truth):
    """比较预测答案和参考答案"""
    if pred_answer is None or ground_truth is None:
        return False
    
    # 将答案转换为字符串并标准化
    pred_str = str(pred_answer).strip().lower()
    truth_str = str(ground_truth).strip().lower()
    
    # 检查是否是数字答案
    try:
        pred_num = float(pred_str.replace(',', ''))
        truth_num = float(truth_str.replace(',', ''))
        # 允许小的误差
        return abs(pred_num - truth_num) < 0.01
    except (ValueError, TypeError):
        pass
    
    # 对于描述性答案，检查是否包含关键信息
    # 这里使用简单的子字符串匹配，可以根据需要扩展为更复杂的逻辑
    return pred_str in truth_str or truth_str in pred_str

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="测试Qwen模型在不同数据集上的表现")
    parser.add_argument('--dataset', type=str, default='CLEVR', choices=['CLEVR', 'SAT'], 
                        help="选择测试的数据集: CLEVR 或 SAT")
    parser.add_argument('--samples', type=int, default=20, help="要测试的样本数量")
    parser.add_argument('--batch_size', type=int, default=8, help="批量推理的大小")
    parser.add_argument('--threshold', type=int, default=2, help="小误差的阈值")
    parser.add_argument('--counting_only', action='store_true', 
                        help="仅测试计数题")
    args = parser.parse_args()
    
    # 1. 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")

    # 确保文件夹存在
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 设置环境变量，强制使用本地缓存
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    os.environ["HF_HOME"] = models_dir
    os.environ["HF_HUB_CACHE"] = models_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_DATASETS_CACHE"] = data_dir

    # 2. 加载模型
    print("正在加载SpaceQwen视觉模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "remyxai/SpaceQwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("remyxai/SpaceQwen2.5-VL-3B-Instruct")
    
    # 3. 加载数据集
    if args.dataset == "CLEVR":
        print("正在加载CLEVR数据集...")
        dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB", split="train", cache_dir=data_dir)
        print(f"数据集加载完成，共有 {len(dataset)} 个样本")

        # CLEVR批量推理的处理函数
        def process_batch(batch):
            # 构造messages
            messages_list = []
            for sample in batch:
                image = sample['image']
                question = sample['problem']
                # 明确要求输出<answer>标签
                prompt = f"{question} Please answer with the answer, wrapped in <answer> </answer> tags, e.g. <answer> 3 </answer>."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages_list.append(messages)
            return messages_list

    elif args.dataset == "SAT":
        print("正在加载SAT数据集...")
        # 尝试从本地文件加载
        sat_local_path = os.path.join(data_dir, "SAT", "SAT_train_15000.json")
        if os.path.exists(sat_local_path):
            print(f"从本地文件加载SAT数据集: {sat_local_path}")
            with open(sat_local_path, "r", encoding="utf-8") as f:
                sat_data = json.load(f)
            
            # 创建一个简单的数据集格式，保持一致的接口
            class SimpleDataset:
                def __init__(self, data):
                    self.data = data
                    
                def __len__(self):
                    return len(self.data)
                    
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = SimpleDataset(sat_data)
            print(f"数据集加载完成，共有 {len(dataset)} 个样本")
        else:
            # 如果找不到本地文件，则尝试从Hugging Face加载
            print("未找到本地SAT数据集，尝试从Hugging Face加载...")
            dataset = load_dataset("array/SAT", split="train", cache_dir=data_dir)
            print(f"数据集加载完成，共有 {len(dataset)} 个样本")
        
        # SAT批量推理的处理函数
        def process_batch(batch):
            # 处理SAT数据集样本
            processed_samples = []
            
            for sample in batch:
                # 检查是否为本地文件格式
                if 'messages' in sample and 'images' in sample:
                    # 本地文件格式
                    user_message = sample['messages'][0]['content']
                    if "<image>" in user_message:
                        user_message = user_message.replace("<image>", "").strip()
                    
                    ground_truth = sample['messages'][1]['content']
                    
                    # 加载图像
                    image_paths = sample['images']
                    if not image_paths:
                        print(f"样本没有图像路径，跳过")
                        continue
                        
                    try:
                        image_path = os.path.join(data_dir, "SAT", image_paths[0])
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        print(f"无法加载图像 {image_path}: {e}")
                        # 尝试使用相对路径
                        try:
                            image = Image.open(image_paths[0]).convert("RGB")
                        except:
                            print(f"无法加载图像，跳过")
                            continue
                    
                    # 检查是否为计数题（通过solution判断是否包含数字）
                    is_counting = False
                    number_in_solution = extract_number_from_answer(ground_truth)
                    if number_in_solution is not None:
                        is_counting = True
                    
                    # 移除跳过非计数题的逻辑，确保所有题目都会被处理
                    # 处理后的样本
                    processed_sample = {
                        'problem': user_message,
                        'solution': f"<answer>{ground_truth}</answer>",
                        'image': image,
                        'is_counting': is_counting,  # 标记是否为计数题
                        'number_answer': number_in_solution  # 存储计数答案
                    }
                    
                else:
                    # Hugging Face格式
                    processed_sample = process_sat_sample(sample)
                    
                    # 移除跳过非计数题的逻辑，确保所有题目都会被处理
                
                if processed_sample:
                    processed_samples.append(processed_sample)
            
            # 构造messages
            messages_list = []
            for sample in processed_samples:
                image = sample['image']
                question = sample['problem']
                # 明确要求输出<answer>标签
                prompt_suffix = ""
                if sample.get('is_counting', False):
                    prompt_suffix = " Please make sure to include the numeric answer."
                
                prompt = f"{question}{prompt_suffix} Please respond with the answer wrapped in <answer> </answer> tags, e.g. <answer>3</answer>; <answer>42</answer>; <answer>7.5</answer>."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                messages_list.append(messages)
            return messages_list, processed_samples

    # 4. 批量推理参数
    batch_size = args.batch_size  # 一次推理的样本数
    max_samples = min(args.samples, len(dataset))  # 总共推理多少个样本
    small_diff_threshold = args.threshold

    results = []
    
    # 统计变量
    total_samples = 0
    invalid_answers = 0
    counting_questions = 0
    correct_counting = 0     # 完全正确的计数题数量
    correct_within_threshold = 0  # 在阈值范围内的计数题数量
    wrong_counting = 0
    num_diff_LE_smt = 0
    num_diff_grthan_smt = 0
    other_questions = 0

    print(f"\n开始批量测试 {max_samples} 个样本，每批 {batch_size} ...")

    for batch_start in tqdm(range(0, max_samples, batch_size), desc="批量推理"):
        batch_end = min(batch_start + batch_size, max_samples)
        batch = [dataset[i] for i in range(batch_start, batch_end)]

        # 处理不同数据集的批量推理
        if args.dataset == "CLEVR":
            messages_list = process_batch(batch)
            processed_batch = batch
        elif args.dataset == "SAT":
            messages_list, processed_batch = process_batch(batch)

        # 批量处理chat template和vision info
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        image_inputs_list = []
        video_inputs_list = []
        for m in messages_list:
            image_inputs, video_inputs = process_vision_info(m)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)

        # 判断是否所有 video_inputs 都是 None
        if all(v is None for v in video_inputs_list):
            inputs = processor(
                text=texts,
                images=image_inputs_list,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(
                text=texts,
                images=image_inputs_list,
                videos=video_inputs_list,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 保存结果
        for idx, (sample, output_text) in enumerate(zip(processed_batch, output_texts)):
            global_idx = batch_start + idx
            
            # 显示结果
            print(f"\n样本 {global_idx + 1}:")
            print(f"问题: {sample['problem']}")
            print(f"参考答案: {sample['solution']}")
            print(f"模型回答: {output_text}")
            
            # 评估回答
            current_sample, question_type, is_correct = evaluate_response(
                output_text, sample, args.dataset, small_diff_threshold, global_idx
            )
            results.append(current_sample)
            
            # 更新统计
            total_samples += 1
            
            if args.dataset == "SAT":
                if question_type == "invalid":
                    invalid_answers += 1
                elif question_type == "counting":
                    counting_questions += 1
                    if current_sample.get("completely_correct", False):
                        correct_counting += 1
                        correct_within_threshold += 1  # 完全正确的也在阈值范围内
                    elif current_sample.get("correct", False):  # 使用current_sample中的correct属性
                        correct_within_threshold += 1  # 在阈值范围内但不完全正确
                        wrong_counting += 1  # 不完全正确
                        diff = current_sample.get("diff", 0)
                        num_diff_LE_smt += 1  # 差距在阈值内
                    else:
                        wrong_counting += 1
                        diff = current_sample.get("diff", 0)
                        if diff <= small_diff_threshold:
                            num_diff_LE_smt += 1
                        else:
                            num_diff_grthan_smt += 1
                elif question_type == "other":
                    other_questions += 1
            
            # 打印当前批次后的实时统计
            print("-" * 30)
            print(f"当前已处理样本数: {total_samples}")
            print(f"计数问题数: {counting_questions}")
            print(f"其他类型问题数: {other_questions}")
            print(f"无效答案数: {invalid_answers}")
            if counting_questions > 0:
                print(f"计数题完全正确率: {correct_counting / counting_questions:.2%}")
                print(f"计数题阈值内正确率: {correct_within_threshold / counting_questions:.2%}")
                print(f"计数题完全正确数: {correct_counting}")
                print(f"计数题阈值内正确数: {correct_within_threshold}")
                print(f"计数题不完全正确数: {wrong_counting}")
                if wrong_counting > 0:
                    print(f"差距小于等于{small_diff_threshold}的计数题: {num_diff_LE_smt}")
                    print(f"差距大于{small_diff_threshold}的计数题: {num_diff_grthan_smt}")
                    small_ratio = num_diff_LE_smt / wrong_counting if wrong_counting > 0 else 0
                    large_ratio = num_diff_grthan_smt / wrong_counting if wrong_counting > 0 else 0
                    print(f"预测误差小于等于{small_diff_threshold}的比例: {small_ratio:.2%}")
                    print(f"预测误差大于{small_diff_threshold}的比例: {large_ratio:.2%}")
            
            print("=" * 50)

    # 5. 保存全部结果
    if args.dataset == "CLEVR":
        final_filename = os.path.join(results_dir, f"clevr_qwen_batch_results_{max_samples}_smt{small_diff_threshold}.json")
        accuracy_stats = {
            "total_samples": total_samples,
            "invalid_answers": invalid_answers,
            "counting_questions": counting_questions,
            "correct_counting": correct_counting,
            "wrong_counting": wrong_counting,
            "small_diff_errors": num_diff_LE_smt,
            "large_diff_errors": num_diff_grthan_smt,
            "counting_accuracy": correct_counting / counting_questions if counting_questions > 0 else 0,
            "threshold": small_diff_threshold,
        }
                
        print(f"\n统计结果:")
        print("-" * 50)
        print(f"总样本数: {total_samples}")
        print(f"无效答案数: {invalid_answers}")
        print(f"计数问题数: {counting_questions}")
        if counting_questions > 0:
            print(f"计数题准确率: {accuracy_stats['counting_accuracy']:.2%}")
            print(f"计数题完全正确数: {correct_counting}")
            print(f"计数题错误数: {wrong_counting}")
            if wrong_counting > 0:
                print(f"差距小于等于{small_diff_threshold}的计数题: {num_diff_LE_smt}")
                print(f"差距大于{small_diff_threshold}的计数题: {num_diff_grthan_smt}")
            
    elif args.dataset == "SAT":
        final_filename = os.path.join(results_dir, f"sat_qwen_batch_results_{max_samples}_smt{small_diff_threshold}.json")
        
        # 计算计数题准确率
        exact_accuracy = correct_counting / counting_questions if counting_questions > 0 else 0
        threshold_accuracy = correct_within_threshold / counting_questions if counting_questions > 0 else 0
        
        if wrong_counting > 0:
            ratio_small_diff = num_diff_LE_smt / wrong_counting
            ratio_large_diff = num_diff_grthan_smt / wrong_counting
        else:
            ratio_small_diff = 0
            ratio_large_diff = 0
        
        accuracy_stats = {
            "total_samples": total_samples,
            "counting_questions": counting_questions,
            "other_questions": other_questions,
            "invalid_answers": invalid_answers,
            "exact_correct_counting": correct_counting,
            "threshold_correct_counting": correct_within_threshold,
            "wrong_counting": wrong_counting,
            "small_diff_errors": num_diff_LE_smt,
            "large_diff_errors": num_diff_grthan_smt,
            "exact_counting_accuracy": exact_accuracy,
            "threshold_counting_accuracy": threshold_accuracy,
            "ratio_small_diff": ratio_small_diff,
            "ratio_large_diff": ratio_large_diff,
            "threshold": small_diff_threshold,
        }
        
        print(f"\n统计结果:")
        print("-" * 50)
        print(f"总样本数: {total_samples}")
        print(f"计数问题数: {counting_questions}")
        print(f"其他类型问题数: {other_questions}")
        print(f"无效答案数: {invalid_answers}")
        if counting_questions > 0:
            print(f"计数题完全正确率: {exact_accuracy:.2%}")
            print(f"计数题阈值内正确率: {threshold_accuracy:.2%}")
            print(f"计数题完全正确数: {correct_counting}")
            print(f"计数题阈值内正确数: {correct_within_threshold}")
            print(f"计数题不完全正确数: {wrong_counting}")
            if wrong_counting > 0:
                print(f"差距小于等于{small_diff_threshold}的计数题: {num_diff_LE_smt}")
                print(f"差距大于{small_diff_threshold}的计数题: {num_diff_grthan_smt}")
                print(f"预测误差小于等于{small_diff_threshold}的比例: {ratio_small_diff:.2%}")
                print(f"预测误差大于{small_diff_threshold}的比例: {ratio_large_diff:.2%}")
            
    # 将统计数据添加到结果中
    results_with_stats = {
        "statistics": accuracy_stats,
        "results": results
    }

    with open(final_filename, "w", encoding="utf-8") as f:
        json.dump(results_with_stats, f, ensure_ascii=False, indent=2)

    print(f"\n已将完整结果保存到: {final_filename}")
    print("-" * 50) 

if __name__ == "__main__":
    main() 