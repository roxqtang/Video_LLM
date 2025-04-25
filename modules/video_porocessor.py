import torch
import torch.nn as nn

class VideoProcessor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        temporal_downsample_factor: int = 1,
    ):
        super().__init__()
        self.temporal_downsample_factor = temporal_downsample_factor
        
        # 视频帧间的 3D 卷积（深度为 2）
        self.temporal_conv = nn.Conv3d(
            embed_dim, 
            embed_dim, 
            kernel_size=(2, 1, 1),  # 时间深度为 2
            stride=(temporal_downsample_factor, 1, 1),
            padding=(0, 0, 0),
        )
        
    def forward(self, x, num_frames, h, w):
        # x: [B, N, C] - 视频 tokens
        # N = num_frames * h * w
        B, N, C = x.shape
        
        # 将特征图重塑为 [B, T, H, W, C]
        x = x.view(B, num_frames, h, w, C)
        
        # 应用 3D 时空卷积
        # 先转换为卷积需要的格式
        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        
        # 确保帧数足够用于 3D 卷积
        if num_frames > 1:
            # 使用滑动窗口处理成对的帧
            x_processed = []
            for i in range(0, num_frames - 1, self.temporal_downsample_factor):
                # 选择当前时间窗口
                x_window = x[:, :, i:i+2]
                
                # 只处理完整的窗口
                if x_window.shape[2] == 2:
                    x_conv = self.temporal_conv(x_window)  # [B, C, 1, H, W]
                    x_processed.append(x_conv)
            
            # 如果有处理后的帧
            if x_processed:
                x = torch.cat(x_processed, dim=2)  # [B, C, T', H, W]
            else:
                # 如果无法应用 3D 卷积，保持原样
                x = x[:, :, :1]  # 只保留第一帧
        else:
            # 单帧情况，复制处理
            x = x.repeat(1, 1, 2, 1, 1)  # 复制帧
            x = self.temporal_conv(x)  # [B, C, 1, H, W]
        
        # 转回原始格式
        x = x.permute(0, 2, 3, 4, 1)  # [B, T', H, W, C]
        
        # 展平空间和时间维度
        new_num_frames = x.shape[1]
        x = x.reshape(B, -1, C)  # [B, T'*H*W, C]
        
        return x, new_num_frames
