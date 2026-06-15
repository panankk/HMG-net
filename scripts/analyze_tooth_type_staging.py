#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_tooth_type_staging.py

Purpose
-------
统计专家正畸轨迹中，不同牙齿类型（incisor / canine / premolar / molar）
是否呈现不同的阶段性运动模式。

核心思想类似 test.py：
1. 读取每个病例的逐步专家轨迹 JSON。
2. 计算相邻 step 之间每颗牙的 translation displacement 和 rotation change。
3. 根据移动阈值判断每颗牙在每一步是否 active。
4. 按四种牙齿类型分组统计 activation probability。
5. 将每个病例的治疗过程归一化为若干 phase，例如 early / middle / late。
6. 输出每个 phase 中四类牙齿的活跃概率，用于支持“不同类型牙齿存在阶段性运动模式”的论文分析。

Input format
------------
默认假设每个病例目录下包含若干 step JSON 文件，例如：
case_id/
    step_0.json
    step_1.json
    ...
每个 JSON 格式默认类似：
{
    "11": [x, y, z, ...],
    "12": [x, y, z, ...],
    ...
}
其中前三个数为 translation。若后续包含四元数 qx,qy,qz,qw，则可同时统计 rotation；否则 rotation 会自动跳过。

Usage
-----
python analyze_tooth_type_staging.py \
    --expert_root /home/pa/version4/data/test_data \
    --num_cases 1000 \
    --move_thres 0.1 \
    --rot_thres_deg 0.8 \
    --num_phases 3 \
    --output_csv tooth_type_staging_stats.csv
"""

import os
import re
import json
import glob
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


# 32-slot FDI order used in your project.
FDI_IDS = [
    18, 17, 16, 15, 14, 13, 12, 11,
    21, 22, 23, 24, 25, 26, 27, 28,
    48, 47, 46, 45, 44, 43, 42, 41,
    31, 32, 33, 34, 35, 36, 37, 38
]


def tooth_type_from_fdi(fdi: int) -> str:
    """Map FDI tooth number to four coarse tooth types."""
    unit = fdi % 10
    if unit in (1, 2):
        return "incisor"
    if unit == 3:
        return "canine"
    if unit in (4, 5):
        return "premolar"
    if unit in (6, 7, 8):
        return "molar"
    return "unknown"


TOOTH_TYPES = ["incisor", "canine", "premolar", "molar"]


def parse_step(filename: str) -> int:
    """Extract step index from a filename such as step_000.json or xxx_step12.json."""
    base = os.path.basename(filename)
    match = re.search(r"step_?(\d+)", base)
    return int(match.group(1)) if match else -1


def normalize_quat(q: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return None
    return q / norm


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute angular difference between two quaternions in degrees.
    Assumes q format [qx, qy, qz, qw].
    Does not require scipy.
    """
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    if q1 is None or q2 is None:
        return 0.0

    # Relative rotation angle can be computed from absolute dot product:
    # angle = 2 * arccos(|dot(q1, q2)|)
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    angle = 2.0 * np.arccos(dot)
    return float(np.degrees(angle))


def load_case_steps(case_dir: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Load all step JSON files from one case.

    Returns
    -------
    positions: [T, 32, 3]
    quats: [T, 32, 4] or None
    valid_mask: [32]
    """
    files = sorted(glob.glob(os.path.join(case_dir, "*step*.json")), key=parse_step)
    if len(files) < 2:
        raise ValueError("Need at least two step files.")

    T = len(files)
    positions = np.zeros((T, 32, 3), dtype=np.float32)
    quats = np.zeros((T, 32, 4), dtype=np.float32)
    has_quat_any = False
    valid_mask = np.zeros(32, dtype=bool)

    for t, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fdi_str, vals in data.items():
            try:
                fdi = int(fdi_str)
            except Exception:
                continue

            if fdi not in FDI_IDS:
                continue

            idx = FDI_IDS.index(fdi)
            vals = list(vals)
            if len(vals) < 3:
                continue

            positions[t, idx] = np.asarray(vals[:3], dtype=np.float32)
            valid_mask[idx] = True

            # Optional quaternion: [x,y,z,qx,qy,qz,qw] or longer.
            if len(vals) >= 7:
                q = np.asarray(vals[3:7], dtype=np.float32)
                if np.linalg.norm(q) > 1e-8:
                    quats[t, idx] = q
                    has_quat_any = True

    return positions, (quats if has_quat_any else None), valid_mask


def split_indices_by_phase(num_steps: int, num_phases: int) -> List[np.ndarray]:
    """
    Split step indices [0, num_steps-1] into equal normalized treatment phases.
    Here num_steps means number of transitions, i.e., T-1.
    """
    idx = np.arange(num_steps)
    return np.array_split(idx, num_phases)


def analyze_case(
    case_dir: str,
    move_thres: float,
    rot_thres_deg: float,
    num_phases: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Analyze one case and return activation probabilities per phase and tooth type.

    Returns dict:
    {
      "trans": [num_phases, 4],
      "rot": [num_phases, 4],
      "either": [num_phases, 4],
      "counts": [num_phases, 4]  # denominator counts
    }
    """
    try:
        positions, quats, valid_mask = load_case_steps(case_dir)
    except Exception:
        return None

    num_transitions = positions.shape[0] - 1
    if num_transitions <= 0:
        return None

    # Translation displacement: [T-1, 32]
    d_pos = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    active_trans = d_pos > move_thres

    # Rotation displacement: [T-1, 32]
    if quats is not None:
        d_rot = np.zeros((num_transitions, 32), dtype=np.float32)
        for t in range(num_transitions):
            for i in range(32):
                if valid_mask[i]:
                    d_rot[t, i] = quat_angle_deg(quats[t, i], quats[t + 1, i])
        active_rot = d_rot > rot_thres_deg
    else:
        active_rot = np.zeros_like(active_trans, dtype=bool)

    active_either = active_trans | active_rot

    phases = split_indices_by_phase(num_transitions, num_phases)

    out_trans = np.zeros((num_phases, len(TOOTH_TYPES)), dtype=np.float32)
    out_rot = np.zeros_like(out_trans)
    out_either = np.zeros_like(out_trans)
    out_counts = np.zeros_like(out_trans)

    for p, phase_idx in enumerate(phases):
        if len(phase_idx) == 0:
            continue

        for type_idx, type_name in enumerate(TOOTH_TYPES):
            tooth_indices = [
                i for i, fdi in enumerate(FDI_IDS)
                if tooth_type_from_fdi(fdi) == type_name and valid_mask[i]
            ]
            if not tooth_indices:
                continue

            # Denominator: phase steps × valid teeth in this type.
            denom = len(phase_idx) * len(tooth_indices)
            out_counts[p, type_idx] = denom

            out_trans[p, type_idx] = active_trans[np.ix_(phase_idx, tooth_indices)].mean()
            out_rot[p, type_idx] = active_rot[np.ix_(phase_idx, tooth_indices)].mean()
            out_either[p, type_idx] = active_either[np.ix_(phase_idx, tooth_indices)].mean()

    return {
        "trans": out_trans,
        "rot": out_rot,
        "either": out_either,
        "counts": out_counts,
    }


def weighted_average(case_results: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    """
    Weighted average across cases using valid tooth-step counts as denominator.
    """
    numerator = None
    denominator = None

    for res in case_results:
        vals = res[key]
        counts = res["counts"]

        if numerator is None:
            numerator = vals * counts
            denominator = counts.copy()
        else:
            numerator += vals * counts
            denominator += counts

    return numerator / np.maximum(denominator, 1e-8)


def save_csv(path: str, stats: Dict[str, np.ndarray], num_phases: int) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric",
            "phase",
            "phase_range",
            "tooth_type",
            "activation_probability",
        ])

        for metric, arr in stats.items():
            for p in range(num_phases):
                start = p / num_phases
                end = (p + 1) / num_phases
                phase_name = f"Phase {p+1}"
                phase_range = f"{start:.2f}-{end:.2f}"
                for type_idx, type_name in enumerate(TOOTH_TYPES):
                    writer.writerow([
                        metric,
                        phase_name,
                        phase_range,
                        type_name,
                        f"{arr[p, type_idx]:.6f}",
                    ])


def print_table(stats: Dict[str, np.ndarray], metric: str, num_phases: int) -> None:
    arr = stats[metric]
    print("\n" + "=" * 72)
    print(f"{metric.upper()} activation probability by tooth type and treatment phase")
    print("=" * 72)
    header = "Phase".ljust(16) + "".join([t.ljust(14) for t in TOOTH_TYPES])
    print(header)
    print("-" * 72)

    for p in range(num_phases):
        phase_label = f"{int(100*p/num_phases)}-{int(100*(p+1)/num_phases)}%"
        row = phase_label.ljust(16)
        for type_idx in range(len(TOOTH_TYPES)):
            row += f"{arr[p, type_idx]*100:>6.2f}%".ljust(14)
        print(row)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_root", type=str, required=True,
                        help="Root directory containing expert case folders.")
    parser.add_argument("--num_cases", type=int, default=1000,
                        help="Number of cases to sample. Use -1 for all cases.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move_thres", type=float, default=0.1,
                        help="Translation threshold in mm for active movement.")
    parser.add_argument("--rot_thres_deg", type=float, default=0.8,
                        help="Rotation threshold in degrees for active movement when quaternion is available.")
    parser.add_argument("--num_phases", type=int, default=3,
                        help="Number of normalized treatment phases.")
    parser.add_argument("--output_csv", type=str, default="tooth_type_staging_stats.csv")
    args = parser.parse_args()

    if not os.path.exists(args.expert_root):
        raise FileNotFoundError(f"expert_root does not exist: {args.expert_root}")

    case_ids = sorted([
        d for d in os.listdir(args.expert_root)
        if os.path.isdir(os.path.join(args.expert_root, d))
    ])

    if args.num_cases > 0 and args.num_cases < len(case_ids):
        random.seed(args.seed)
        case_ids = random.sample(case_ids, args.num_cases)

    print(f"Analyzing {len(case_ids)} expert cases...")
    print(f"Translation threshold: {args.move_thres} mm")
    print(f"Rotation threshold: {args.rot_thres_deg} deg")
    print(f"Number of phases: {args.num_phases}")

    case_results = []
    for case_id in tqdm(case_ids):
        res = analyze_case(
            os.path.join(args.expert_root, case_id),
            move_thres=args.move_thres,
            rot_thres_deg=args.rot_thres_deg,
            num_phases=args.num_phases,
        )
        if res is not None:
            case_results.append(res)

    if not case_results:
        print("No valid cases found.")
        return

    stats = {
        "translation": weighted_average(case_results, "trans"),
        "rotation": weighted_average(case_results, "rot"),
        "either": weighted_average(case_results, "either"),
    }

    print(f"\nValid analyzed cases: {len(case_results)}")
    print_table(stats, "translation", args.num_phases)
    print_table(stats, "rotation", args.num_phases)
    print_table(stats, "either", args.num_phases)

    save_csv(args.output_csv, stats, args.num_phases)
    print(f"\nSaved CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
