# ==========================================
# 1. 导入 Backbone 组件
# ==========================================
from .backbone.pointnet import MiniPointNet
# 消融实验：空 Backbone
from .backbone.identity import IdentityBackbone

# ==========================================
# 2. 导入 Head 组件
# ==========================================
# 核心预测头：不确定性回归
from .head.uncertain import UncertainRegressionHead
# 消融实验：确定性预测头
from .head.deterministic import DeterministicHead
# 消融实验：空 Head
from .head.identity import IdentityHead

# ==========================================
# 3. 导入 Strategy 组件
# ==========================================
# 核心门控机制：双流解耦
from .strategy.dual_gate import AdvancedDualStreamMaskHead
# 消融实验：直通门控
from .strategy.identity import IdentityGate
from .strategy.single_gate import SingleStreamGate


# ==========================================
# 4. 导出列表 (Export List)
# ==========================================
__all__ = [
    # Backbone
    'MiniPointNet',
    'IdentityBackbone',
    
    # Head
    'UncertainRegressionHead',
    'DeterministicHead',
    'IdentityHead',
    
    # Strategy
    'AdvancedDualStreamMaskHead',
    'IdentityGate',
    'SingleStreamGate',
]
