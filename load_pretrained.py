import torch
import torch.nn as nn
import os
import json
from safetensors.torch import load_file
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from myqwen import Qwen25VLModel

class WeightLoader:
    def __init__(self, model_path: str):
        """
        初始化权重加载器
        
        Args:
            model_path: 包含预训练权重的文件夹路径
        """
        self.model_path = model_path
        self.index_file = os.path.join(model_path, "model.safetensors.index.json")
        self.weight_files = {}
        self.weight_map = {}
        
        # 加载索引文件
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                index_data = json.load(f)
                self.weight_map = index_data.get("weight_map", {})
        else:
            print(f"无法找到索引文件: {self.index_file}")
            return
        
        # 获取权重文件的路径
        weight_files = set(self.weight_map.values())
        for file in weight_files:
            self.weight_files[file] = os.path.join(model_path, file)
            if not os.path.exists(self.weight_files[file]):
                print(f"找不到权重文件: {self.weight_files[file]}")
    
    def load_config(self) -> dict:
        """加载模型配置"""
        config_path = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            print(f"找不到配置文件: {config_path}")
            return {}
    
    def _create_mapping_rules(self) -> List[Tuple[str, str]]:
        """
        创建预训练权重和自定义模型之间的映射规则
        
        Returns:
            映射规则列表，每个规则是一个(源前缀, 目标前缀)元组
        """
        # 定义映射规则 - 从预训练权重架构到自定义模型架构
        mapping_rules = [
            # 视觉编码器部分
            ("visual.patch_embed", "visual_encoder.patch_embed"),
            ("visual.blocks", "visual_encoder.blocks"),
            ("visual.merger", "visual_encoder.compress"),
            
            # LLM部分
            ("model.embed_tokens", "token_embeddings"),
            ("model.layers", "llm_blocks"),
            ("model.norm", "llm_norm"),
            
            # 视觉投影
            ("visual.merger.mlp", "vision_proj"),
            
            # 特殊token嵌入
            ("vision_start_embedding", "vision_start_token"),
            ("vision_end_embedding", "vision_end_token"),
            
            # 输出头
            ("lm_head", "lm_head"),
        ]
        
        return mapping_rules
    
    def _map_weight_name(self, original_name: str) -> Optional[str]:
        """
        根据映射规则转换权重名称
        
        Args:
            original_name: 原始权重名称
            
        Returns:
            映射后的权重名称，如果无法映射则返回None
        """
        mapping_rules = self._create_mapping_rules()
        
        # 对每个规则尝试映射
        for source_prefix, target_prefix in mapping_rules:
            if original_name.startswith(source_prefix):
                # 替换前缀
                mapped_name = original_name.replace(source_prefix, target_prefix, 1)
                return mapped_name
        
        # 如果没有匹配的规则
        return None
    
    def load_weights(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        """
        将预训练权重加载到自定义模型中
        
        Args:
            model: 自定义模型实例
            
        Returns:
            加载了权重的模型和详细的加载报告
        """
        # 收集所有需要加载的权重
        weights_to_load = OrderedDict()
        loaded_weights = []
        missing_weights = []
        unused_weights = []
        shape_mismatch_report = []
        
        # 遍历所有权重名称
        for original_name, file_name in self.weight_map.items():
            mapped_name = self._map_weight_name(original_name)
            
            if mapped_name is not None:
                # 检查模型中是否存在该参数
                try:
                    param = self._get_parameter_from_model(model, mapped_name)
                    if param is not None:
                        weights_to_load[original_name] = (mapped_name, file_name)
                        loaded_weights.append(mapped_name)
                    else:
                        missing_weights.append((mapped_name, "参数在模型中不存在"))
                except (AttributeError, KeyError) as e:
                    missing_weights.append((mapped_name, f"访问错误: {str(e)}"))
            else:
                unused_weights.append(original_name)
        
        # 加载权重
        for original_name, (mapped_name, file_name) in weights_to_load.items():
            try:
                # 加载特定权重文件
                weight_file = self.weight_files[file_name]
                weights = load_file(weight_file, device="cpu")
                
                if original_name in weights:
                    weight_value = weights[original_name]
                    param = self._get_parameter_from_model(model, mapped_name)
                    
                    # 检查形状是否匹配
                    if param.shape == weight_value.shape:
                        param.data.copy_(weight_value)
                    else:
                        error_msg = f"形状不匹配: {mapped_name}, 模型形状: {param.shape}, 权重形状: {weight_value.shape}"
                        print(error_msg)
                        shape_mismatch_report.append((original_name, mapped_name, str(param.shape), str(weight_value.shape)))
                        missing_weights.append((mapped_name, "形状不匹配"))
                        if mapped_name in loaded_weights:
                            loaded_weights.remove(mapped_name)
            except Exception as e:
                error_msg = f"加载权重时出错 {original_name} -> {mapped_name}: {str(e)}"
                print(error_msg)
                missing_weights.append((mapped_name, str(e)))
                if mapped_name in loaded_weights:
                    loaded_weights.remove(mapped_name)
        
        # 创建详细报告
        load_report = {
            "成功加载": len(loaded_weights),
            "总权重数": len(loaded_weights) + len([m for m, _ in missing_weights]),
            "形状不匹配列表": shape_mismatch_report,
            "缺失权重列表": missing_weights,
            "未使用的原始权重数": len(unused_weights),
            "未使用的原始权重": unused_weights[:10] + ['...'] if len(unused_weights) > 10 else unused_weights
        }
        
        print(f"成功加载 {load_report['成功加载']}/{load_report['总权重数']} 权重")
        print(f"未使用的原始权重: {len(unused_weights)}")
        
        return model, load_report
    
    def _get_parameter_from_model(self, model: nn.Module, param_name: str) -> Optional[torch.nn.Parameter]:
        """
        从模型中获取参数
        
        Args:
            model: 模型实例
            param_name: 参数名称，如 "visual_encoder.blocks.0.norm1.weight"
            
        Returns:
            参数对象，如果不存在则返回None
        """
        components = param_name.split('.')
        current = model
        
        # 逐层访问参数
        for i, comp in enumerate(components):
            if hasattr(current, comp):
                current = getattr(current, comp)
            else:
                try:
                    # 尝试作为索引访问（对于ModuleList）
                    if comp.isdigit() and isinstance(current, nn.ModuleList):
                        current = current[int(comp)]
                    else:
                        return None
                except (IndexError, TypeError):
                    return None
        
        # 确保返回的是参数
        if isinstance(current, nn.Parameter):
            return current
        return None


def main():
    # 模型权重路径
    model_path = "models/models--remyxai--SpaceQwen2.5-VL-3B-Instruct/snapshots/ff71e2f363d635f5dbbd655bf11ac7f976f94c7d"
    
    # 加载器实例
    loader = WeightLoader(model_path)
    
    # 加载配置
    config = loader.load_config()
    print("加载模型配置:")
    print(f"- 隐藏维度: {config.get('hidden_size', 2048)}")
    print(f"- 层数: {config.get('num_hidden_layers', 36)}")
    print(f"- 注意力头数: {config.get('num_attention_heads', 16)}")
    
    # 根据配置初始化模型
    model = Qwen25VLModel(
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=config.get("vision_config", {}).get("hidden_size", 1280),
        vision_depth=32,  # ViT层数
        vision_num_heads=16,
        vision_mlp_ratio=2.67,  # 修正为3420/1280=2.67
        window_size=8,
        llm_dim=config.get("hidden_size", 2048),
        llm_depth=config.get("num_hidden_layers", 36),
        llm_num_heads=config.get("num_attention_heads", 16),
        llm_mlp_ratio=float(config.get("intermediate_size", 11008) / config.get("hidden_size", 2048)),
        max_position_embeddings=config.get("max_position_embeddings", 128000),
        vocab_size=config.get("vocab_size", 151936),
    )
    
    print(f"\n创建Qwen2.5-VL模型实例:")
    print(f"- 架构: 视觉编码器 + 多模态旋转位置编码 + 视频处理器 + LLM")
    
    # 加载权重
    print("\n开始加载预训练权重...")
    model, load_report = loader.load_weights(model)
    
    # 打印详细的加载报告
    print("\n===== 权重加载详细报告 =====")
    print(f"成功加载: {load_report['成功加载']}/{load_report['总权重数']} 权重")
    
    # 打印形状不匹配的详细信息
    if load_report['形状不匹配列表']:
        print("\n形状不匹配的权重:")
        for original, mapped, model_shape, weight_shape in load_report['形状不匹配列表']:
            print(f"  - 原始名称: {original}")
            print(f"    映射名称: {mapped}")
            print(f"    模型形状: {model_shape}, 权重形状: {weight_shape}")
    
    # 打印缺失权重的详细信息
    if load_report['缺失权重列表']:
        print("\n缺失的权重:")
        for mapped_name, reason in load_report['缺失权重列表']:
            print(f"  - {mapped_name}: {reason}")
    
    # 保存详细报告到文件
    with open("weight_loading_report.json", "w", encoding="utf-8") as f:
        json.dump(load_report, f, ensure_ascii=False, indent=2)
    print("\n详细加载报告已保存到: weight_loading_report.json")
    
    # 保存具有权重的模型
    output_path = "pretrained_myqwen.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\n模型已保存到: {output_path}")
    
    # 设置为评估模式
    model.eval()
    print("\n模型加载完成，已设置为评估模式")
    
    return model


if __name__ == "__main__":
    main() 