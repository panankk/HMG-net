import os
import json
import numpy as np
import torch
import open3d as o3d  # 🔥 替换 trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 全局配置
# ==========================================
# 路径配置
RAW_DATA_ROOT = "/path/to/test_data"
PROCESSED_ROOT = "path/to/test_data_processed"
# Hull 输出路径 (改为 open3d 生成的 ply)
HULL_OUTPUT_ROOT = "path/to/test_data_hulls" 
REMOVE_JSON_PATH = "path/to/remove_idx_summary.json"

# 牙位定义
FDI_IDS = [
    18, 17, 16, 15, 14, 13, 12, 11,
    21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41,
    31, 32, 33, 34, 35, 36, 37, 38
]
FDI_TO_IDX = {fdi: i for i, fdi in enumerate(FDI_IDS)}

# 采样参数
HULL_SAMPLE_POINTS = 5000 
FEAT_SAMPLE_POINTS = 1024  

# ==========================================
# 2. 辅助函数
# ==========================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_case(case_id, case_path, remove_dict):
    processed_case_dir = os.path.join(PROCESSED_ROOT, case_id)
    hull_case_dir = os.path.join(HULL_OUTPUT_ROOT, case_id)
    ensure_dir(processed_case_dir)
    ensure_dir(hull_case_dir)
    
    # --- A. 解析轨迹 (Poses) ---
    # 逻辑保持不变：读取 stepX.txt
    step_files = sorted(
        [f for f in os.listdir(case_path) if f.startswith('step') and f.endswith('.txt')], 
        key=lambda x: int(x.replace('step', '').replace('.txt', ''))
    )
    
    if len(step_files) == 0:
        return

    T = len(step_files)
    poses_9d = torch.zeros((T, 32, 9))
    teeth_mask = torch.zeros(32)

    for t, step_file in enumerate(step_files):
        file_path = os.path.join(case_path, step_file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 8: continue
            try:
                fdi = int(parts[0])
            except ValueError:
                continue
            if fdi not in FDI_TO_IDX: continue 
            idx = FDI_TO_IDX[fdi]
            
           
            if t == 0: teeth_mask[idx] = 1.0
            
            # 解析位姿
            pos = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            quat = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            
            r = R.from_quat(quat)
            mat = r.as_matrix()
            rot_6d = np.concatenate([mat[:, 0], mat[:, 1]])
            
            poses_9d[t, idx, :3] = torch.from_numpy(pos)
            poses_9d[t, idx, 3:9] = torch.from_numpy(rot_6d)

    #  应用拔牙掩码 (Remove IDs)
    if case_id in remove_dict:
        remove_fdis = remove_dict[case_id]
        for r_fdi in remove_fdis:
            if r_fdi in FDI_TO_IDX:
                r_idx = FDI_TO_IDX[r_fdi]
                teeth_mask[r_idx] = 0.0 # 标记为拔除
    
    # --- B. 处理几何 ---
    shape_features = torch.zeros((32, FEAT_SAMPLE_POINTS, 3)) 
    models_dir = os.path.join(case_path, "models")
    
    # 禁止输出 open3d 的读取日志
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    for fdi, idx in FDI_TO_IDX.items():
        if teeth_mask[idx] == 0: continue 
        
        # 匹配文件名
        possible_names = [f"{fdi}._Root.stl", f"{fdi}_Root.stl", f"{fdi}.stl"]
        stl_path = None
        for name in possible_names:
            p = os.path.join(models_dir, name)
            if os.path.exists(p):
                stl_path = p
                break
        
        if stl_path:
            try:
                # 1. 读取 Mesh (Open3D)
                mesh = o3d.io.read_triangle_mesh(stl_path)
                
                # --- 分支 1: 生成高精度 Hull (用于物理碰撞检测) ---
                # 遵循作者逻辑: 5000点均匀采样 -> 凸包
                pcd_hull = mesh.sample_points_uniformly(number_of_points=HULL_SAMPLE_POINTS)
                hull, _ = pcd_hull.compute_convex_hull()
                
                # 保存 Hull (.ply) 用于后续 FCL 碰撞检测
                hull_save_path = os.path.join(hull_case_dir, f"{case_id}_{fdi}.ply")
                o3d.io.write_triangle_mesh(hull_save_path, hull)
                
                # --- 分支 2: 生成特征点云 (用于 PointNet 输入) ---
                # 采样 1024 点
                pcd_feat = mesh.sample_points_uniformly(number_of_points=FEAT_SAMPLE_POINTS)
                points = np.asarray(pcd_feat.points) # [1024, 3]
                
                # 归一化 (去中心化)
                centroid = points.mean(axis=0)
                points_centered = points - centroid
                shape_features[idx] = torch.from_numpy(points_centered).float()
                
            except Exception as e:
                # print(f"  [Error] Processing {stl_path}: {e}")
                pass

    # --- C. 计算 Meta Info ---
    start_pos = poses_9d[0, :, :3]
    end_pos = poses_9d[-1, :, :3]
    dist_per_tooth = torch.norm(start_pos - end_pos, dim=-1)
    
    # 仅计算存在的牙齿 (mask=1)
    valid_dist = dist_per_tooth * teeth_mask
    num_valid = teeth_mask.sum()
    
    if num_valid > 0:
        initial_diff = valid_dist.sum() / num_valid
    else:
        initial_diff = torch.tensor(1.0)

    # --- D. 保存 ---
    torch.save(poses_9d, os.path.join(processed_case_dir, 'poses_9d.pt'))
    torch.save(shape_features, os.path.join(processed_case_dir, 'shape_feature.pt'))
    
    meta = {
        'initial_diff': initial_diff.item(),
        'mask': teeth_mask
    }
    torch.save(meta, os.path.join(processed_case_dir, 'meta.pt'))

# ==========================================
# 3. 主执行入口
# ==========================================
def run_preprocess():
    ensure_dir(PROCESSED_ROOT)
    ensure_dir(HULL_OUTPUT_ROOT)
    
    # 加载 Remove Summary
    if os.path.exists(REMOVE_JSON_PATH):
        with open(REMOVE_JSON_PATH, 'r') as f:
            remove_dict = json.load(f)
        print(f" Loaded Remove Summary: {len(remove_dict)} cases.")
    else:
        print(" Warning: remove_idx_summary.json not found!")
        remove_dict = {}

    # 获取所有病例
    case_ids = sorted([d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))])
    
    print(f" Preprocess (Open3D Hull 5k) Start...")
    print(f"   - Hull Output: {HULL_OUTPUT_ROOT}")
    print(f"   - Feat Output: {PROCESSED_ROOT}")
    
    for case_id in tqdm(case_ids):
        process_case(case_id, os.path.join(RAW_DATA_ROOT, case_id), remove_dict)
        
    print("\n All Done!")

if __name__ == "__main__":
    run_preprocess()
