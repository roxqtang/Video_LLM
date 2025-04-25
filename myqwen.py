import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.vision_encoder import VisualEncoder, RMSNorm, TransformerLayer
from modules.video_porocessor import VideoProcessor
from modules.m_rope import MRoPE


class Qwen25VLModel(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        vision_depth: int = 24,
        vision_num_heads: int = 16,
        vision_mlp_ratio: float = 4.0,
        window_size: int = 8,
        llm_dim: int = 4096,
        llm_depth: int = 32,
        llm_num_heads: int = 32,
        llm_mlp_ratio: float = 4.0,
        max_position_embeddings: int = 4096,
        vocab_size: int = 32000,
    ):
        super().__init__()
        
        # 视觉编码器
        self.visual_encoder = VisualEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=vision_num_heads,
            mlp_ratio=vision_mlp_ratio,
            window_size=window_size,
        )
        
        # 视频处理器
        self.video_processor = VideoProcessor(
            embed_dim=embed_dim,
            temporal_downsample_factor=1,  # 无时间下采样
        )
        
        # 视觉投影层，将视觉特征映射到 LLM 维度
        #self.vision_proj = nn.Linear(embed_dim, llm_dim, bias=False)
        self.vision_proj = nn.Linear(1280, 2048, bias=False)
        
        # 特殊 token 嵌入
        self.vision_start_token = nn.Parameter(torch.zeros(1, 1, llm_dim))
        self.vision_end_token = nn.Parameter(torch.zeros(1, 1, llm_dim))
        
        # 多模态旋转位置编码
        self.m_rope = MRoPE(llm_dim // llm_num_heads, max_position_embeddings)
        
        # 文本嵌入层
        self.token_embeddings = nn.Embedding(vocab_size, llm_dim)
        
        # LLM Transformer 块（简化版，实际中会更复杂）
        self.llm_blocks = nn.ModuleList([
            TransformerLayer(
                dim=llm_dim,
                num_heads=llm_num_heads,
                mlp_ratio=llm_mlp_ratio,
                qkv_bias=True,
                use_window_attn=False,  # LLM 部分通常不使用窗口注意力
                block_name="llm",
            ) for _ in range(llm_depth)
        ])
        
        self.llm_norm = RMSNorm(llm_dim)
        
        # 输出头
        self.lm_head = nn.Linear(llm_dim, vocab_size, bias=False)
        
    def _prepare_vision_input(self, pixel_values, is_video=False):
        batch_size = pixel_values.shape[0]
        
        if is_video:
            # 视频输入: [B, T, C, H, W]
            B, T, C, H, W = pixel_values.shape
            
            # 展平 batch 和时间维度处理
            pixel_values = pixel_values.reshape(B * T, C, H, W)
            
            # 编码每一帧
            vision_features = self.visual_encoder(pixel_values)  # [B*T, N, C]
            
            # 重塑回带时间维度的格式
            _, N, C = vision_features.shape
            vision_features = vision_features.reshape(B, T, N, C)
            
            # 将所有帧合并为单个序列
            vision_features = vision_features.reshape(B, T * N, C)
            
            # 应用视频处理器
            h = w = int(math.sqrt(N))
            vision_features, new_num_frames = self.video_processor(vision_features, T, h, w)
            
            # 生成位置 IDs
            # 时间 ID 随每帧递增
            time_ids = torch.repeat_interleave(
                torch.arange(new_num_frames, device=vision_features.device), 
                h * w
            ).unsqueeze(0).expand(B, -1)
            
            # 高度和宽度 ID 基于每个 token 在每帧图像中的位置
            y_pos = torch.arange(h, device=vision_features.device)
            x_pos = torch.arange(w, device=vision_features.device)
            grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
            
            height_ids = grid_y.flatten().unsqueeze(0).expand(new_num_frames, -1).reshape(-1).unsqueeze(0).expand(B, -1)
            width_ids = grid_x.flatten().unsqueeze(0).expand(new_num_frames, -1).reshape(-1).unsqueeze(0).expand(B, -1)
        else:
            # 图像输入: [B, C, H, W]
            vision_features = self.visual_encoder(pixel_values)  # [B, N, C]
            
            # 图像视为两个相同的帧
            _, N, C = vision_features.shape
            h = w = int(math.sqrt(N))
            
            # 生成位置 IDs
            # 图像的时间 ID 保持恒定
            time_ids = torch.zeros(N, device=vision_features.device).unsqueeze(0).expand(batch_size, -1)
            
            # 高度和宽度 ID 基于每个 token 在图像中的位置
            y_pos = torch.arange(h, device=vision_features.device)
            x_pos = torch.arange(w, device=vision_features.device)
            grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
            
            height_ids = grid_y.flatten().unsqueeze(0).expand(batch_size, -1)
            width_ids = grid_x.flatten().unsqueeze(0).expand(batch_size, -1)
            
        # 对 visual features 应用投影，将维度从 embed_dim 映射到 llm_dim
        vision_features = self.vision_proj(vision_features)
        
        # 添加视觉标记 tokens
        vision_start_tokens = self.vision_start_token.expand(batch_size, -1, -1)
        vision_end_tokens = self.vision_end_token.expand(batch_size, -1, -1)
        
        vision_features = torch.cat([vision_start_tokens, vision_features, vision_end_tokens], dim=1)
        
        # 更新位置 IDs 以包含视觉标记 tokens
        # 为开始和结束 token 使用特殊的 ID (-1, -1) 表示它们不是真正的视觉 tokens
        special_time_ids = torch.full((batch_size, 1), -1, device=time_ids.device)
        time_ids = torch.cat([special_time_ids, time_ids, special_time_ids], dim=1)
        
        special_height_width_ids = torch.full((batch_size, 1), -1, device=height_ids.device)
        height_ids = torch.cat([special_height_width_ids, height_ids, special_height_width_ids], dim=1)
        width_ids = torch.cat([special_height_width_ids, width_ids, special_height_width_ids], dim=1)
        
        return vision_features, time_ids, height_ids, width_ids
    
    def forward(
        self,
        input_ids=None,
        vision_input=None,
        is_video=False,
        attention_mask=None,
        position_ids=None,
        labels=None,
    ):
        batch_size = input_ids.shape[0] if input_ids is not None else vision_input.shape[0]
        
        # 处理视觉输入
        vision_embeds = None
        vision_time_ids = None
        vision_height_ids = None
        vision_width_ids = None
        
        if vision_input is not None:
            vision_embeds, vision_time_ids, vision_height_ids, vision_width_ids = self._prepare_vision_input(
                vision_input, is_video=is_video
            )
        
        # 处理文本输入
        if input_ids is not None:
            # 获取文本嵌入
            text_embeds = self.token_embeddings(input_ids)
            
            # 合并文本和视觉嵌入（如果有视觉输入）
            if vision_embeds is not None:
                # 合并嵌入
                hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
                
                # 合并位置 IDs
                if position_ids is None:
                    # 生成文本的位置 ID，从视觉序列长度开始
                    seq_length = input_ids.shape[1]
                    vision_length = vision_embeds.shape[1]
                    position_ids = torch.arange(
                        vision_length, vision_length + seq_length, 
                        device=input_ids.device
                    ).unsqueeze(0).expand(batch_size, -1)
                
                # 文本的时间、高度和宽度 ID 都相同（等于其位置 ID）
                text_time_ids = position_ids
                text_height_ids = position_ids
                text_width_ids = position_ids
                
                # 合并视觉和文本的位置 ID
                time_ids = torch.cat([vision_time_ids, text_time_ids], dim=1)
                height_ids = torch.cat([vision_height_ids, text_height_ids], dim=1)
                width_ids = torch.cat([vision_width_ids, text_width_ids], dim=1)
            else:
                # 只有文本输入
                hidden_states = text_embeds
                
                # 生成文本的位置 ID
                if position_ids is None:
                    seq_length = input_ids.shape[1]
                    position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                
                # 文本的时间、高度和宽度 ID 都相同（等于其位置 ID）
                time_ids = position_ids
                height_ids = position_ids
                width_ids = position_ids
        else:
            # 只有视觉输入（少见情况）
            hidden_states = vision_embeds
            time_ids = vision_time_ids
            height_ids = vision_height_ids
            width_ids = vision_width_ids
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 扩展注意力掩码以适应视觉 tokens（如果有）
            if vision_embeds is not None and input_ids is not None:
                vision_length = vision_embeds.shape[1]
                vision_attention_mask = torch.ones(
                    (batch_size, vision_length), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([vision_attention_mask, attention_mask], dim=1)
        
        # 通过 LLM Transformer 块
        for i, block in enumerate(self.llm_blocks):
            hidden_states = block(hidden_states)
        
        # 最终 Layer Norm
        hidden_states = self.llm_norm(hidden_states)
        
        # 计算语言建模损失（如果提供标签）
        loss = None
        logits = None
        
        if input_ids is not None:
            # 使用 LM 头获取 logits
            if vision_embeds is not None:
                # 只对文本部分应用 LM 头
                vision_length = vision_embeds.shape[1]
                text_hidden_states = hidden_states[:, vision_length:]
                logits = self.lm_head(text_hidden_states)
            else:
                # 所有隐藏状态都是文本
                logits = self.lm_head(hidden_states)
            
            # 计算损失（如果提供标签）
            if labels is not None:
                # 移位标签并计算交叉熵损失
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.lm_head.out_features), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits, "hidden_states": hidden_states}
    
    def generate(
        self,
        input_ids,
        vision_input=None,
        is_video=False,
        attention_mask=None,
        position_ids=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        eos_token_id=None,
        **kwargs
    ):
        """
        使用自回归生成回答文本
        """
        batch_size = input_ids.shape[0]
        
        # 初始化处理视觉输入（如果有）
        vision_embeds = None
        vision_time_ids = None
        vision_height_ids = None
        vision_width_ids = None
        
        if vision_input is not None:
            vision_embeds, vision_time_ids, vision_height_ids, vision_width_ids = self._prepare_vision_input(
                vision_input, is_video=is_video
            )
        
        # 追踪生成的 token
        generated_ids = input_ids
        
        for _ in range(max_length):
            # 准备当前输入
            curr_input_ids = generated_ids
            
            # 生成位置 ID
            curr_seq_len = curr_input_ids.shape[1]
            if position_ids is None:
                curr_position_ids = torch.arange(curr_seq_len, device=curr_input_ids.device).unsqueeze(0).expand(batch_size, -1)
            else:
                # 使用提供的位置 ID 并扩展以适应新 token
                orig_len = position_ids.shape[1]
                if curr_seq_len > orig_len:
                    # 追加新的位置 ID
                    new_position_ids = torch.arange(orig_len, curr_seq_len, device=position_ids.device).unsqueeze(0).expand(batch_size, -1)
                    curr_position_ids = torch.cat([position_ids, new_position_ids], dim=1)
                else:
                    curr_position_ids = position_ids
            
            # 获取文本嵌入
            text_embeds = self.token_embeddings(curr_input_ids)
            
            # 合并文本和视觉嵌入（如果有）
            if vision_embeds is not None:
                hidden_states = torch.cat([vision_embeds, text_embeds], dim=1)
                
                # 视觉序列长度
                vision_length = vision_embeds.shape[1]
                
                # 更新文本位置 ID
                text_time_ids = curr_position_ids
                text_height_ids = curr_position_ids
                text_width_ids = curr_position_ids
                
                # 合并视觉和文本位置 ID
                time_ids = torch.cat([vision_time_ids, text_time_ids], dim=1)
                height_ids = torch.cat([vision_height_ids, text_height_ids], dim=1)
                width_ids = torch.cat([vision_width_ids, text_width_ids], dim=1)
            else:
                hidden_states = text_embeds
                time_ids = curr_position_ids
                height_ids = curr_position_ids
                width_ids = curr_position_ids
            
            # 应用注意力掩码（如果有）
            if attention_mask is not None:
                # 扩展注意力掩码以适应新 token 和视觉 tokens（如果有）
                curr_seq_len = curr_input_ids.shape[1]
                orig_len = attention_mask.shape[1]
                
                if curr_seq_len > orig_len:
                    # 追加新的注意力掩码位
                    new_attn_mask = torch.ones((batch_size, curr_seq_len - orig_len), dtype=attention_mask.dtype, device=attention_mask.device)
                    curr_attention_mask = torch.cat([attention_mask, new_attn_mask], dim=1)
                else:
                    curr_attention_mask = attention_mask
                
                if vision_embeds is not None:
                    # 添加视觉注意力掩码
                    vision_length = vision_embeds.shape[1]
                    vision_attention_mask = torch.ones((batch_size, vision_length), dtype=curr_attention_mask.dtype, device=curr_attention_mask.device)
                    curr_attention_mask = torch.cat([vision_attention_mask, curr_attention_mask], dim=1)
            
            # 通过 LLM Transformer 块
            for block in self.llm_blocks:
                hidden_states = block(hidden_states)
            
            # 最终 Layer Norm
            hidden_states = self.llm_norm(hidden_states)
            
            # 获取下一个 token 的 logits
            if vision_embeds is not None:
                # 只使用文本的最后一个 token
                next_token_logits = self.lm_head(hidden_states[:, -1:])
            else:
                next_token_logits = self.lm_head(hidden_states[:, -1:])
            
            next_token_logits = next_token_logits.squeeze(1)
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 应用重复惩罚
            if repetition_penalty > 1.0:
                for i in range(batch_size):
                    for prev_token in set(generated_ids[i].tolist()):
                        # 如果 token 已经出现，降低其概率
                        next_token_logits[i, prev_token] /= repetition_penalty
            
            # 应用 top-k 过滤
            if top_k > 0:
                # 过滤掉概率低的 token
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 应用 top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 移位使第一个 token 超过阈值的索引保持
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # 获取下一个 token
            if do_sample:
                # 对 logits 应用 softmax 并采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 追加生成的 token
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            
            # 检查是否生成了 EOS token
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
        
        return generated_ids
