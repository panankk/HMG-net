import torch
import os
import numpy as np
from torch.utils.data import Dataset


RAW_ERROR_CASES = [
]

EXISTING_BAD_CASES = [
]

NEW_BAD_CASES = [
]

BLACKLIST_IDS = {case_name.replace('.json', '') for case_name in RAW_ERROR_CASES}
FINAL_BAD_CASES = set(EXISTING_BAD_CASES) | set(NEW_BAD_CASES)

# 0: 切牙，1: 尖牙，2: 前磨牙，3: 磨牙
TOOTH_TYPE_MAP = {
    11:0, 12:0, 21:0, 22:0, 31:0, 32:0, 41:0, 42:0, 
    13:1, 23:1, 33:1, 43:1,                         
    14:2, 15:2, 24:2, 25:2, 34:2, 35:2, 44:2, 45:2, 
    16:3, 17:3, 18:3, 26:3, 27:3, 28:3,             
    36:3, 37:3, 38:3, 46:3, 47:3, 48:3
}

FDI_LIST = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28, 
            48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


class OrthoDataset(Dataset):
    def __init__(self, processed_root, window_size=1):
        self.processed_root = processed_root
        self.window_size = window_size
        
        if not os.path.exists(processed_root):
             raise ValueError(f"Data root {processed_root} does not exist!")
        
        all_dirs = sorted(os.listdir(processed_root))
        
        self.cases = [
            d for d in all_dirs 
            if d not in BLACKLIST_IDS and d not in FINAL_BAD_CASES 
            and os.path.isdir(os.path.join(processed_root, d))
        ]
        print(f"🧹 V39 Strategic Dataset (Temporal Window: {self.window_size}): {len(self.cases)} Cases Loaded.")

    def __len__(self):
        return len(self.cases)

    @staticmethod
    def make_combined_state(curr_p, goal_p, teeth_mask):
        is_batch = (curr_p.dim() == 3)
        cp = curr_p if is_batch else curr_p.unsqueeze(0)
        gp = goal_p if is_batch else goal_p.unsqueeze(0)
        tm = teeth_mask if is_batch else teeth_mask.unsqueeze(0)

        curr_pos = cp[..., :3]
        mask_expand = tm.unsqueeze(-1)
        sum_pos = (curr_pos * mask_expand).sum(dim=1, keepdim=True)
        count = mask_expand.sum(dim=1, keepdim=True) + 1e-6
        centroid = sum_pos / count
        
        local_pos = curr_pos - centroid 
        diff_pos = gp[..., :3] - cp[..., :3]
        state_pos = torch.cat([local_pos, diff_pos], dim=-1)

        curr_rot = cp[..., 3:9]
        goal_rot = gp[..., 3:9]
        diff_rot = goal_rot - curr_rot 
        state_rot = torch.cat([curr_rot, diff_rot], dim=-1)
        
        state = torch.cat([state_pos, state_rot], dim=-1)
        return state.squeeze(0) if not is_batch else state


    @staticmethod
    def compute_gate_masks_at_step(poses_9d, t):
        curr_p = poses_9d[t]
        next_p = poses_9d[t + 1]

        if t == 0:
            real_prev_pos_active = torch.zeros(32, dtype=torch.bool)
            real_prev_rot_active = torch.zeros(32, dtype=torch.bool)
        else:
            prev_p = poses_9d[t - 1]
            prev_dist_pos = torch.norm(curr_p[..., :3] - prev_p[..., :3], dim=-1)
            real_prev_pos_active = prev_dist_pos > 0.05

            prev_dist_rot = torch.norm(curr_p[..., 3:9] - prev_p[..., 3:9], dim=-1)
            real_prev_rot_active = prev_dist_rot > 0.01

        target_pos_delta = next_p[..., :3] - curr_p[..., :3]
        step_dist_pos = torch.norm(target_pos_delta, dim=-1)

        pos_thresh = torch.where(
            real_prev_pos_active,
            torch.tensor(0.03),  # Keep
            torch.tensor(0.1)    # Start
        )
        gt_mask_pos = (step_dist_pos > pos_thresh).float()

        target_rot_delta_raw = next_p[..., 3:9] - curr_p[..., 3:9]
        step_dist_rot = torch.norm(target_rot_delta_raw, dim=-1)

        rot_thresh = torch.where(
            real_prev_rot_active,
            torch.tensor(0.01),
            torch.tensor(0.02)
        )
        gt_mask_rot = (step_dist_rot > rot_thresh).float()

        return gt_mask_pos, gt_mask_rot

    def compute_transition_scores(self, poses_9d, teeth_mask):
        """
        计算每个候选 step 的 transition score。
        score(t) = 平移 gate 状态切换数 + 旋转 gate 状态切换数。
        对 t=0，使用当前 active 数量作为初始启动 transition。
        """
        T = poses_9d.shape[0]
        max_t = T - 1  # valid t: [0, T-2]
        scores = []

        prev_pos_mask = None
        prev_rot_mask = None
        valid_mask = teeth_mask.float()

        for step_t in range(max_t):
            gt_mask_pos, gt_mask_rot = self.compute_gate_masks_at_step(poses_9d, step_t)
            gt_mask_pos = gt_mask_pos * valid_mask
            gt_mask_rot = gt_mask_rot * valid_mask

            if step_t == 0:
                transition_pos = gt_mask_pos.sum()
                transition_rot = gt_mask_rot.sum()
            else:
                transition_pos = torch.abs(gt_mask_pos - prev_pos_mask).sum()
                transition_rot = torch.abs(gt_mask_rot - prev_rot_mask).sum()

            scores.append(transition_pos + transition_rot)
            prev_pos_mask = gt_mask_pos.clone()
            prev_rot_mask = gt_mask_rot.clone()

        if len(scores) == 0:
            return torch.zeros(0)

        return torch.stack(scores).float()

    def sample_timestep(self, poses_9d, teeth_mask):
        T = poses_9d.shape[0]
        max_t = T - 1  # valid t: [0, T-2], because next_p = poses_9d[t+1]

        if max_t <= 1:
            return 0

        # 90%: keep the original uniform sampling.
        if torch.rand(1).item() < 0.9:
            return torch.randint(0, max_t, (1,)).item()

        # 10%: sample from transition-aware hard steps.
        scores = self.compute_transition_scores(poses_9d, teeth_mask)
        if scores.numel() != max_t:
            return torch.randint(0, max_t, (1,)).item()

        hard_indices = torch.nonzero(scores >= 4.0, as_tuple=False).view(-1)

        # If a case has no hard transition step, fall back to uniform sampling.
        if hard_indices.numel() == 0:
            return torch.randint(0, max_t, (1,)).item()

        select_idx = torch.randint(0, hard_indices.numel(), (1,)).item()
        return int(hard_indices[select_idx].item())

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        case_path = os.path.join(self.processed_root, case_id)
        
        try:
            poses_9d = torch.load(os.path.join(case_path, 'poses_9d.pt'), map_location='cpu', weights_only=True)
            shape_emb = torch.load(os.path.join(case_path, 'shape_feature.pt'), map_location='cpu', weights_only=True)
            meta = torch.load(os.path.join(case_path, 'meta.pt'), map_location='cpu', weights_only=True)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.cases))

        teeth_mask = meta['mask']
        T = poses_9d.shape[0]
        
      
        t = self.sample_timestep(poses_9d, teeth_mask)

      
        history_indices = [max(0, t - i) for i in range(self.window_size - 1, -1, -1)]
        history_poses = poses_9d[history_indices] # 形状: [W, 32, 9]

       
        curr_p = poses_9d[t]    # 当前状态
        next_p = poses_9d[t+1]  # 下一步状态 (Target)
        goal_p = poses_9d[-1]   # 最终目标状态
        
        # ===========================================================
        # 1. 物理层计算 (Physical Layer - 严格对齐当前 t)
        # ===========================================================
        # A. 基础残差 (Residual)
        diff_pos_vec = goal_p[..., :3] - curr_p[..., :3]
        res_pos = torch.norm(diff_pos_vec, dim=-1) # [32]
        
        diff_rot_vec = goal_p[..., 3:9] - curr_p[..., 3:9]
        res_rot = torch.norm(diff_rot_vec, dim=-1) # [32]
        
        # B. 真实惯性 (Real Inertia - 用于生成 Label)
        if t == 0:
            real_prev_pos_active = torch.zeros(32, dtype=torch.bool)
            real_prev_rot_active = torch.zeros(32, dtype=torch.bool)
        else:
            prev_p = poses_9d[t-1]
            prev_dist_pos = torch.norm(curr_p[..., :3] - prev_p[..., :3], dim=-1)
            real_prev_pos_active = prev_dist_pos > 0.05
            
            prev_dist_rot = torch.norm(curr_p[..., 3:9] - prev_p[..., 3:9], dim=-1)
            real_prev_rot_active = prev_dist_rot > 0.01

        # C. 生成标签 (GT Masks - Schmitt Trigger)
        target_pos_delta = next_p[..., :3] - curr_p[..., :3]
        step_dist_pos = torch.norm(target_pos_delta, dim=-1)
        
        pos_thresh = torch.where(real_prev_pos_active, 
                                 torch.tensor(0.03), # Keep
                                 torch.tensor(0.1))  # Start
        gt_mask_pos = (step_dist_pos > pos_thresh).float()
        
        target_rot_delta_raw = next_p[..., 3:9] - curr_p[..., 3:9]
        step_dist_rot = torch.norm(target_rot_delta_raw, dim=-1)
        
        rot_thresh = torch.where(real_prev_rot_active,
                                 torch.tensor(0.01), 
                                 torch.tensor(0.02))
        gt_mask_rot = (step_dist_rot > rot_thresh).float()

        # ===========================================================
        # 2. 战略特征工程 
        # ===========================================================
        
        # --- A. 身份  ---
        type_ids_np = np.array([TOOTH_TYPE_MAP.get(fdi, 3) for fdi in FDI_LIST])
        type_ids = torch.tensor(type_ids_np, dtype=torch.long)
        
        feat_my_type = torch.zeros(32, 4)
        feat_my_type.scatter_(1, type_ids.unsqueeze(1), 1.0) # [32, 4] One-Hot
        
        # --- B. 环境信号  ---
        is_finished_pos = (res_pos < 0.2).float() 
        is_finished_rot = (res_rot < 0.05).float()
        
        group_rates_pos = []
        group_rates_rot = []
        
        for type_idx in range(4): # Inc, Can, Pre, Mol
            group_mask = (type_ids == type_idx).float()
            total_in_group = group_mask.sum() + 1e-6
            
            rate_pos = (is_finished_pos * group_mask).sum() / total_in_group
            rate_rot = (is_finished_rot * group_mask).sum() / total_in_group
            
            group_rates_pos.append(rate_pos)
            group_rates_rot.append(rate_rot)
            
        feat_group_pos = torch.stack(group_rates_pos).unsqueeze(0).expand(32, 4)
        feat_group_rot = torch.stack(group_rates_rot).unsqueeze(0).expand(32, 4)
        
        # --- C. 惯性特征  ---
        feat_prev_active_pos = real_prev_pos_active.float()
        feat_prev_active_rot = real_prev_rot_active.float()
        
        if torch.rand(1).item() < 0.3: 
            drop_mask = (torch.rand(32) < 0.5) 
            feat_prev_active_pos = feat_prev_active_pos * (1 - drop_mask.float() * gt_mask_pos)
            
        if torch.rand(1).item() < 0.3:
            drop_mask = (torch.rand(32) < 0.5)
            feat_prev_active_rot = feat_prev_active_rot * (1 - drop_mask.float() * gt_mask_rot)

        # ===========================================================
        # 3. 组装返回数据 (Strict Decoupling)
        # ===========================================================
        
       
        history_states = []
        for i in range(self.window_size):
            hist_state = self.make_combined_state(history_poses[i], goal_p, teeth_mask)
            history_states.append(hist_state)
        
        # 堆叠成时序序列张量 [W, 32, 18]
        states_seq = torch.stack(history_states) 
        
        # 组装战略向量 (依然只基于当前帧，完全独立于时序机制)
        strat_vec_pos = torch.cat([
            feat_my_type,                       # [32, 4]
            feat_group_pos,                     # [32, 4]
            res_pos.unsqueeze(-1),              # [32, 1]
            feat_prev_active_pos.unsqueeze(-1)  # [32, 1]
        ], dim=-1)

        strat_vec_rot = torch.cat([
            feat_my_type,                       # [32, 4]
            feat_group_rot,                     # [32, 4]
            res_rot.unsqueeze(-1),              # [32, 1]
            feat_prev_active_rot.unsqueeze(-1)  # [32, 1]
        ], dim=-1)

        return {
            'shape': shape_emb.view(32, 1024, 3),
            
          
            # 形状：[W, 32*18]
            # 当 W=1 时，形状 [1, 576] 与原版完全等价
            'input_seq': states_seq.view(self.window_size, -1),
            
           
            'strat_vec_pos': strat_vec_pos.view(1, -1), 
            'strat_vec_rot': strat_vec_rot.view(1, -1), 
            
           
            'feat_prev_pos': feat_prev_active_pos.view(1, -1),
            'feat_prev_rot': feat_prev_active_rot.view(1, -1),
            'gt_pos_mu': target_pos_delta.view(1, -1),       
            'gt_rot_mu': (target_rot_delta_raw * 100.0).view(1, -1), 
            'gt_mask_pos': gt_mask_pos.view(1, -1),
            'gt_mask_rot': gt_mask_rot.view(1, -1),
            'timestep': torch.tensor([float(t)]),     
            'teeth_mask': teeth_mask.view(1, -1),
            'tooth_types': type_ids 
        }
