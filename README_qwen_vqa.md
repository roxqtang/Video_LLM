# Qwen视觉模型CLEVR数据集测试指南

本项目提供了三个脚本，用于测试和评估Qwen视觉语言模型在CLEVR数据集上的性能。

## 环境准备

确保已安装必要的依赖包：

```bash
pip install transformers datasets pillow tqdm numpy scikit-learn
```

## 文件说明

1. **test_qwen_vqa.py** - 简单测试脚本，运行少量样本进行基本测试
2. **test_qwen_vqa_batch.py** - 批量测试脚本，支持大规模数据处理和结果保存
3. **qwen_vqa_evaluation.py** - 性能评估脚本，计算准确率等指标并详细分析结果

## 使用方法

### 1. 简单测试

用于快速验证模型和数据集是否正常工作：

```bash
python test_qwen_vqa.py
```

默认测试5个样本，可以通过修改代码中的`num_samples`变量来调整。

### 2. 批量测试

适用于对大量数据进行处理：

```bash
python test_qwen_vqa_batch.py
```

主要参数（可自行传入不同参数）：
- `--dataset`: 选择数据集 可以是 `SAT`, `CLEVR`...
- `samples`: 最大处理样本数，默认为20

结果将保存在`results`目录下。

### 3. 模型评估

对模型性能进行评估：

```bash
   python test_spaceqwen_vqa_batch.py --dataset SAT --samples 20
```

主要参数（可在代码中修改）：
- `num_eval_samples`: 评估样本数量，默认为10
- `random_seed`: 随机种子，用于重复实验，默认为42

评估结果将保存在`evaluation_results`目录下。

## 数据和模型缓存

所有脚本均配置为使用本地缓存：

- 模型文件缓存在项目目录下的`models`文件夹
- 数据集缓存在项目目录下的`data`文件夹

这确保了文件不会存储在默认的Hugging Face缓存位置。

## 建议运行顺序

1. 先运行`test_qwen_vqa.py`，确认模型和数据集加载正常
2. 调整`qwen_vqa_evaluation.py`中的样本数量，进行小规模评估
3. 根据需要调整`test_qwen_vqa_batch.py`的参数，进行大规模测试

## 注意事项

- 首次运行时，会下载模型和数据集，可能需要较长时间
- 视觉模型推理较慢，特别是在CPU上，请耐心等待
- 对于大规模测试，建议使用GPU加速（需添加相应配置） 