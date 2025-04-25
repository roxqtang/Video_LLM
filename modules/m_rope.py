import torch
import torch.nn as nn

class MRoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 初始化旋转位置编码
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
    def _compute_rope(self, positions, dim):
        # 计算旋转位置编码
        # positions: [seq_len]
        # 返回: cos, sin 表示的位置编码
        
        # 截断位置超出范围的部分
        positions = torch.clamp(positions, 0, self.max_position_embeddings - 1)
        
        # 计算频率
        freqs = positions.float().unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        # 计算 sin 和 cos
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # 取前半部分维度作为cos，后半部分作为sin
        return cos, sin
    
    def _apply_rope(self, x, cos, sin, dim):
        # 应用旋转位置编码
        # x: [..., seq_len, dim]
        # cos, sin: [seq_len, dim/2]
        
        # 将张量重塑和转置以便于旋转
        x_reshape = x.view(*x.shape[:-1], -1, 2)
        
        # 偶数和奇数维度
        x_1 = x_reshape[..., 0]
        x_2 = x_reshape[..., 1]
        
        # 应用旋转
        rotated_x_1 = x_1 * cos - x_2 * sin
        rotated_x_2 = x_2 * cos + x_1 * sin
        
        # 合并结果
        rotated_x = torch.stack([rotated_x_1, rotated_x_2], dim=-1)
        rotated_x = rotated_x.view(*x.shape)
        
        return rotated_x
        
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        time_ids=None, 
        height_ids=None, 
        width_ids=None
    ):
        # Qwen2.5-VL M-RoPE
        # q, k: [batch_size, num_heads, seq_len, head_dim]
        # time_ids, height_ids, width_ids: [batch_size, seq_len]
        
        # 默认情况下，文本的位置ID是相同的
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        if time_ids is None:
            # 处理文本: 使所有三个位置IDs相同
            time_ids = torch.arange(seq_len, device=q.device).unsqueeze(0).expand(batch_size, -1)
            height_ids = time_ids
            width_ids = time_ids
        
        # 计算旋转位置编码
        time_cos, time_sin = self._compute_rope(time_ids, head_dim // 3)
        height_cos, height_sin = self._compute_rope(height_ids, head_dim // 3)
        width_cos, width_sin = self._compute_rope(width_ids, head_dim // 3)
        
        # 将 head_dim 分成三个部分
        part_dim = head_dim // 3
        
        # 对时间维度应用 RoPE
        q_time = self._apply_rope(q[..., :part_dim], time_cos, time_sin, part_dim)
        k_time = self._apply_rope(k[..., :part_dim], time_cos, time_sin, part_dim)
        
        # 对高度维度应用 RoPE
        q_height = self._apply_rope(q[..., part_dim:2*part_dim], height_cos, height_sin, part_dim)
        k_height = self._apply_rope(k[..., part_dim:2*part_dim], height_cos, height_sin, part_dim)
        
        # 对宽度维度应用 RoPE
        q_width = self._apply_rope(q[..., 2*part_dim:], width_cos, width_sin, part_dim)
        k_width = self._apply_rope(k[..., 2*part_dim:], width_cos, width_sin, part_dim)
        
        # 合并结果
        q_out = torch.cat([q_time, q_height, q_width], dim=-1)
        k_out = torch.cat([k_time, k_height, k_width], dim=-1)
        
        return q_out, k_out
