import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def create_2d_sincos_pos_emb(h, w, dim):
    # 创建 2D 位置编码
    y_pos = torch.arange(h, dtype=torch.float32)
    x_pos = torch.arange(w, dtype=torch.float32)
    
    # 网格坐标
    grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")
    grid = torch.stack([grid_y, grid_x], dim=-1)
    
    # 展平为 (h*w, 2)
    grid = grid.reshape(-1, 2)
    
    # 创建编码
    emb = get_2d_sincos_pos_emb(grid, dim)
    return emb


def get_2d_sincos_pos_emb(grid, dim):
    # 创建 sin/cos 位置编码 
    half_dim = dim // 2
    theta = torch.arange(half_dim, dtype=torch.float32) / half_dim
    theta = 10000 ** -theta
    
    # 计算位置编码
    pos_emb = grid.unsqueeze(-1) @ theta.unsqueeze(0)
    pos_emb = torch.cat([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
    
    if dim % 2 == 1:
        # 如果维度是奇数，添加一个零列
        pos_emb = torch.cat([pos_emb, torch.zeros_like(pos_emb[:, :1])], dim=-1)
        
    return pos_emb


class WindowAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        window_size: int = 8, 
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x, mask=None):
        # x: [B, H*W, C]
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # 将特征图重塑为窗口
        x = x.view(B, H, W, C)
        
        # 将图像划分为不重叠的窗口
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            
        _, H_pad, W_pad, _ = x.shape
        
        # 窗口划分: [B, num_h, num_w, window_size, window_size, C]
        x = x.view(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, self.window_size * self.window_size, C)
        
        # Multi-head self attention
        qkv = self.qkv(x).reshape(B, -1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, num_windows, num_heads, window_size*window_size, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_windows, num_heads, window_size*window_size, head_dim]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_windows, num_heads, window_size*window_size, window_size*window_size]
        
        if mask is not None:
            # 应用掩码（如果有）
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            
        attn = F.softmax(attn, dim=-1)
        
        # 注意力输出
        x = (attn @ v).transpose(2, 3).reshape(B, -1, self.window_size * self.window_size, C)
        x = self.proj(x)
        
        # 窗口合并回原始形状
        num_windows = x.shape[1]
        num_h = H_pad // self.window_size
        num_w = W_pad // self.window_size
        
        x = x.view(B, num_h, num_w, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)
        
        # 移除填充（如果有）
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
            
        # 展平空间维度
        x = x.view(B, H * W, C)
        return x


class FullAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        
        # 多头自注意力
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # 注意力输出
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.671875,  # 使用修正后的MLP比率
        qkv_bias: bool = True,
        window_size: int = 8,
        use_window_attn: bool = True,
        block_name: str = "vision",
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        # 根据层的类型选择全局注意力或窗口注意力
        if use_window_attn:
            self.attn = WindowAttention(dim, num_heads, window_size, qkv_bias)
        else:
            self.attn = FullAttention(dim, num_heads, qkv_bias)
            
        self.norm2 = RMSNorm(dim)
        
        # SwiGLU FFN - 使用正确的比率
        #hidden_features = int(dim * mlp_ratio) 由于模型mlp形状为(1280,3420)，所以使用3420
        if block_name == "vision":
            hidden_features = 3420
        elif block_name == "llm":
            #hidden_features = int(dim * mlp_ratio)
            hidden_features = 11008
        self.mlp = SwiGLU(dim, hidden_features)
        
    def forward(self, x):
        # 第一个残差连接: 注意力
        x = x + self.attn(self.norm1(x))
        # 第二个残差连接: FFN
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 14, 
        in_chans: int = 3, 
        embed_dim: int = 1280,  # 使用配置中的隐藏维度
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 修正视觉编码器patch embed层
        # 使用3D Conv支持视频输入（包含时间维度）
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=(2, patch_size, patch_size),  # (时间, 高度, 宽度)
            stride=(2, patch_size, patch_size)
        )
        
    def forward(self, x):
        # x: [B, C, H, W] 或 [B, T, C, H, W]
        
        # 支持图像和视频输入
        if len(x.shape) == 4:  # 图像输入 [B, C, H, W]
            B, C, H, W = x.shape
            
            # 将图像转换为单帧视频 [B, 1, C, H, W]
            x = x.unsqueeze(1)
            
            # 预处理为两帧（复制），以便与Conv3D兼容
            x = x.repeat(1, 2, 1, 1, 1)  # [B, 2, C, H, W]
            
            # 应用3D卷积
            x = self.proj(x.permute(0, 2, 1, 3, 4))  # [B, embed_dim, 1, H//patch_size, W//patch_size]
            
            # 重新格式化输出
            x = x.squeeze(2)  # [B, embed_dim, H//patch_size, W//patch_size]
            x = x.flatten(2).transpose(1, 2)  # [B, H//patch_size*W//patch_size, embed_dim]
            
        elif len(x.shape) == 5:  # 视频输入 [B, T, C, H, W]
            B, T, C, H, W = x.shape
            
            # 确保有偶数帧用于3D卷积
            if T % 2 != 0:
                padding = torch.zeros(B, 1, C, H, W, device=x.device)
                x = torch.cat([x, padding], dim=1)
                T += 1
            
            # 应用3D卷积
            x = self.proj(x.permute(0, 2, 1, 3, 4))  # [B, embed_dim, T//2, H//patch_size, W//patch_size]
            
            # 重新格式化输出
            T_out = x.shape[2]
            x = x.permute(0, 2, 3, 4, 1)  # [B, T//2, H//patch_size, W//patch_size, embed_dim]
            x = x.reshape(B, T_out * (H // self.patch_size) * (W // self.patch_size), -1)  # [B, T//2*H//patch_size*W//patch_size, embed_dim]
        
        return x


class VisualEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1280,  # 使用配置中的隐藏维度
        depth: int = 32,  # 使用32层，与预训练模型相匹配
        num_heads: int = 16,
        mlp_ratio: float = 2.671875,  # 使用修正后的MLP比率
        qkv_bias: bool = True,
        window_size: int = 8,
        drop_path_rate: float = 0.1,
        block_name: str = "vision",
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # 动态位置编码 (使用 2D-RoPE 代替传统位置嵌入)
        self.use_rope = True
        
        # 构建 Transformer 层
        # 注意：只有少数几层 (如 4 层) 使用 Full Attention，其余使用 Window Attention
        full_attn_layers = 4
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_window_attn = i >= full_attn_layers  # 前 full_attn_layers 层使用 Full Attention
            layer = TransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                window_size=window_size,
                use_window_attn=use_window_attn,
                block_name='vision'
            )
            self.blocks.append(layer)
            
        self.norm = RMSNorm(embed_dim)
        
        # 视觉表示压缩：将相邻的 2×2 tokens 压缩为一个 token
        self.compress = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def compress_tokens(self, x, h, w):
        # x: [B, N, C]
        B, N, C = x.shape
        
        # 确保可以进行 2x2 压缩
        assert h % 2 == 0 and w % 2 == 0, "高度和宽度必须是偶数以进行 2x2 压缩"
        
        # 将特征图重塑为 [B, H, W, C]
        x = x.view(B, h, w, C)
        
        # 将相邻的 2x2 块合并
        x = x.reshape(B, h // 2, 2, w // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, h // 2, w // 2, 4 * C)
        
        # 应用 MLP 压缩
        x = self.compress(x)  # [B, h//2, w//2, C]
        
        # 展平空间维度
        x = x.reshape(B, -1, C)  # [B, (h//2)*(w//2), C]
        
        return x
        
    def forward(self, x):
        # x: [B, C, H, W] 或 [B, T, C, H, W]
        
        # 应用 patch embedding
        x = self.patch_embed(x)  # [B, (H//patch_size)*(W//patch_size), embed_dim] 或 [B, T//2*(H//patch_size)*(W//patch_size), embed_dim]
        
        # 处理视频或图像
        if len(x.shape) == 3:  # 已经完成了reshape
            B, N, C = x.shape
            h = w = int(math.sqrt(N))  # 假设是正方形
        
        # 应用 Transformer 层
        for block in self.blocks:
            x = block(x)
            
        # 应用最终 Layer Norm
        x = self.norm(x)
        
        # 压缩视觉表示 (可选，根据需求)
        # x = self.compress_tokens(x, h, w)
        
        return x
