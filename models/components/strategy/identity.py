import torch
import torch.nn as nn
from .base import BaseStrategy

# ==========================================
# 1. 消融实验组件：IdentityGate
# ==========================================
class IdentityGate(BaseStrategy):
    """
    用于消融实验的“空”Strategy (No-Op / Placeholder)。
    
    目的：
        验证战略门控机制（Dual Gate）是否有效。
        
    行为：
        忽略输入的战略特征数据，直接返回全 1 的掩码（或接近 1 的 Logits）。
        这意味着“所有牙齿都允许移动”，门控机制失效。
        如果模型性能大幅下降，则证明智能门控是必要的。
    """
   
    # 用来吸收 build_strategy 工厂统一下发的多余参数，防止 TypeError
    def __init__(self, config=None, input_dim=10, static_dim=4, embed_dim=32, **kwargs):
        """
        Args:
            config: 配置对象 (可选)
            input_dim, static_dim, embed_dim: 仅用于工厂接口兼容，实际不使用
        """
        super(IdentityGate, self).__init__(config)

    def forward(self, strat_vec):
        """
        Args:
            strat_vec: [B, N, 10] 输入战略特征（将被忽略）
        
        Returns:
            logits: [B, N] 全大数 Logits
                    Sigmoid(大数) ≈ 1.0，表示“完全允许移动”
        """
        # 获取 Batch (B) 和 Num_Teeth (N)
        if strat_vec.dim() == 3:
            B, N = strat_vec.shape[0], strat_vec.shape[1]
        else:
            # 兼容其他维度情况
            B, N = 1, strat_vec.shape[0]
            
        # 返回与输入设备 (GPU/CPU) 一致的全大数张量
        # 10.0 足够大，Sigmoid(10) ≈ 0.99995，相当于直通
        return torch.full((B, N), 10.0, device=strat_vec.device, dtype=strat_vec.dtype)
