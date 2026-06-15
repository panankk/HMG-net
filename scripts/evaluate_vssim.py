#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import csv
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "EXPERT_ROOT": "/path/to/processed_data",
    "OUTPUT_DIR": "/path/to/output_dir",

    "METHODS": [
        {"name": "HMG-Net", "root": "/path/to/HMG_results", "format": "auto"},
        {"name": "DRL",     "root": "/path/to/DRL_results", "format": "auto"},
        {"name": "TMDM",    "root": "/path/to/TMDM_results", "format": "auto"},
        {"name": "IGWO",    "root": "/path/to/IGWO_results", "format": "auto"},
        {"name": "Neural",  "root": "/path/to/Neural_results", "format": "auto"},
    ],

    "NORM_STEPS": 100,

    # Local window size over [time, tooth]. xyz remains a 3D vector, not a sliding axis.
    "TIME_WIN": 7,
    "TOOTH_WIN": 3,

    # SSIM constants. If None, they are chosen automatically from expected motion scale.
    # Translation field is in mm; rotation field is in degrees.
    "C1_TRANS": None,
    "C2_TRANS": None,
    "C1_ROT": None,
    "C2_ROT": None,

    # Dynamic ranges used for automatic C1/C2:
    # C1=(K1*range)^2, C2=(K2*range)^2
    "TRANS_RANGE_MM": 0.25,
    "ROT_RANGE_DEG": 2.0,
    "K1": 0.01,
    "K2": 0.03,

    # Visualization
    "SAVE_VIS": True,
    "VIS_DIR_NAME": "vf_stksm_similarity_maps",
    "VIS_DPI": 180,
    "VIS_MAX_CASES_PER_METHOD": 8,  # avoid creating thousands of images by default; set -1 for all
    "VIS_CMAP": "viridis",
    "VIS_VMIN": 0.0,   # display remapped similarity in [0,1]
    "VIS_VMAX": 1.0,

    "MAX_CASES": -1,
}

FDI_IDS = [
    18,17,16,15,14,13,12,11,
    21,22,23,24,25,26,27,28,
    48,47,46,45,44,43,42,41,
    31,32,33,34,35,36,37,38,
]


# ============================================================
# IO and trajectory utilities
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_step(filename: str) -> int:
    name = os.path.basename(filename)
    m = re.search(r"step_?(\d+)\.(txt|json)$", name)
    if m:
        return int(m.group(1))
    m = re.search(r"step_?(\d+)", name)
    return int(m.group(1)) if m else -1


def case_dir(root: str, case_id: str) -> str:
    direct = os.path.join(root, case_id)
    if os.path.isdir(direct):
        return direct
    if os.path.isdir(root) and os.path.basename(root.rstrip("/")) == case_id:
        return root
    return direct


def list_step_files(root: str, case_id: str, fmt: str = "auto") -> List[str]:
    cdir = case_dir(root, case_id)
    if not os.path.isdir(cdir):
        return []

    fmt = str(fmt).lower()
    if fmt == "json":
        patterns = ["*step*.json"]
    elif fmt == "txt":
        patterns = ["*step*.txt"]
    else:
        patterns = ["*step*.txt", "*step*.json"]

    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(cdir, p)))
    files = sorted(files, key=parse_step)
    return [f for f in files if parse_step(f) >= 0]


def normalize_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-8)


def load_step_json(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    if not isinstance(data, dict):
        return out

    if "positions" in data or "rotations" in data:
        raise RuntimeError(
            f"{path} looks like a full trajectory JSON. Convert it into step_XXX.txt first."
        )

    for k, v in data.items():
        try:
            fdi = int(k)
            if fdi not in FDI_IDS:
                continue
            arr = np.asarray(v, dtype=np.float64)
            if arr.shape[0] < 7:
                continue
            out[fdi] = (arr[0:3], arr[3:7])  # xyzw
        except Exception:
            continue
    return out


def load_step_txt(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8:
                continue
            try:
                fdi = int(p[0])
                if fdi not in FDI_IDS:
                    continue
                pos = np.array([float(x) for x in p[1:4]], dtype=np.float64)
                quat = np.array([float(x) for x in p[4:8]], dtype=np.float64)  # xyzw
                out[fdi] = (pos, quat)
            except Exception:
                continue
    return out


def load_step(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    return load_step_txt(path) if os.path.splitext(path)[1].lower() == ".txt" else load_step_json(path)


def load_trajectory(root: str, case_id: str, fmt: str = "auto"):
    files = list_step_files(root, case_id, fmt)
    if len(files) < 2:
        return None

    positions = np.zeros((len(files), len(FDI_IDS), 3), dtype=np.float64)
    quats = np.zeros((len(files), len(FDI_IDS), 4), dtype=np.float64)
    quats[:, :, 3] = 1.0
    valid_mask = np.zeros(len(FDI_IDS), dtype=bool)

    for t, fp in enumerate(files):
        step = load_step(fp)
        for fdi, (pos, quat) in step.items():
            idx = FDI_IDS.index(fdi)
            positions[t, idx] = pos
            quats[t, idx] = quat
            if t == 0:
                valid_mask[idx] = True

    return {
        "files": files,
        "positions": positions,
        "quats": normalize_quats(quats),
        "valid_mask": valid_mask,
    }


def compute_translation_vectors(traj: dict) -> np.ndarray:
    return traj["positions"][1:] - traj["positions"][:-1]


def compute_rotation_vectors(traj: dict) -> np.ndarray:
    """
    Return [T-1, 32, 3] relative rotation vectors in degrees.
    ΔR = R_t^{-1} R_{t+1}; scipy outputs radians, converted to degrees.
    """
    q_prev = traj["quats"][:-1].reshape(-1, 4)
    q_next = traj["quats"][1:].reshape(-1, 4)

    r_prev = R.from_quat(q_prev)
    r_next = R.from_quat(q_next)
    r_diff = r_prev.inv() * r_next

    rv_rad = r_diff.as_rotvec().reshape(traj["quats"].shape[0] - 1, len(FDI_IDS), 3)
    return rv_rad * (180.0 / np.pi)


def normalize_time_vector(mat: np.ndarray, norm_steps: int) -> np.ndarray:
    if mat is None or mat.shape[0] == 0:
        return np.zeros((norm_steps, len(FDI_IDS), 3), dtype=np.float64)
    if mat.shape[0] == norm_steps:
        return mat.astype(np.float64)
    return zoom(mat, (norm_steps / mat.shape[0], 1, 1), order=1).astype(np.float64)


# ============================================================
# Vector-valued SSIM / VF-STKSM
# ============================================================
def ssim_constants(cfg: dict, field_type: str):
    if field_type == "trans":
        c1 = cfg.get("C1_TRANS", None)
        c2 = cfg.get("C2_TRANS", None)
        rng = cfg.get("TRANS_RANGE_MM", 0.25)
    elif field_type == "rot":
        c1 = cfg.get("C1_ROT", None)
        c2 = cfg.get("C2_ROT", None)
        rng = cfg.get("ROT_RANGE_DEG", 2.0)
    else:
        raise ValueError(f"Unknown field_type: {field_type}")

    k1 = cfg.get("K1", 0.01)
    k2 = cfg.get("K2", 0.03)
    if c1 is None:
        c1 = (k1 * rng) ** 2
    if c2 is None:
        c2 = (k2 * rng) ** 2
    return float(c1), float(c2)


def pad_field_reflect(field: np.ndarray, time_rad: int, tooth_rad: int) -> np.ndarray:
    return np.pad(
        field,
        pad_width=((time_rad, time_rad), (tooth_rad, tooth_rad), (0, 0)),
        mode="reflect",
    )


def vector_ssim_window(x_win: np.ndarray, y_win: np.ndarray, c1: float, c2: float):
    """
    x_win, y_win: [time_win, tooth_win, 3]
    Treat every local time-tooth location as a 3D vector sample.
    xyz is vector-valued; it is not treated as a scalar sliding dimension.
    """
    x = x_win.reshape(-1, 3).astype(np.float64)
    y = y_win.reshape(-1, 3).astype(np.float64)
    n = x.shape[0]

    mux = x.mean(axis=0)
    muy = y.mean(axis=0)
    xc = x - mux
    yc = y - muy

    if n > 1:
        var_x = float(np.sum(xc * xc) / (n - 1))
        var_y = float(np.sum(yc * yc) / (n - 1))
        cov_xy = float(np.sum(xc * yc) / (n - 1))
    else:
        var_x = float(np.sum(xc * xc))
        var_y = float(np.sum(yc * yc))
        cov_xy = float(np.sum(xc * yc))

    mu_dot = float(np.dot(mux, muy))
    mu_x2 = float(np.dot(mux, mux))
    mu_y2 = float(np.dot(muy, muy))

    luminance = (2.0 * mu_dot + c1) / (mu_x2 + mu_y2 + c1)
    cs = (2.0 * cov_xy + c2) / (var_x + var_y + c2)
    full = luminance * cs

    return float(np.clip(full, -1.0, 1.0)), float(np.clip(cs, -1.0, 1.0))


def vector_ssim_maps(v_method: np.ndarray,
                     v_expert: np.ndarray,
                     valid_teeth: np.ndarray,
                     time_win: int,
                     tooth_win: int,
                     c1: float,
                     c2: float):
    """
    Compute local vector-valued SSIM maps.

    Inputs:
        v_method, v_expert: [T, 32, 3]
        valid_teeth: [32] bool

    Outputs:
        full_map, cs_map: [T, 32]
        Invalid teeth are set to nan and ignored in mean / visualization.
    """
    if time_win % 2 == 0 or tooth_win % 2 == 0:
        raise ValueError("TIME_WIN and TOOTH_WIN must be odd numbers.")

    T, N, C = v_method.shape
    assert C == 3
    assert v_expert.shape == v_method.shape

    tr = time_win // 2
    nr = tooth_win // 2

    pm = pad_field_reflect(v_method, tr, nr)
    pe = pad_field_reflect(v_expert, tr, nr)

    full_map = np.full((T, N), np.nan, dtype=np.float64)
    cs_map = np.full((T, N), np.nan, dtype=np.float64)

    for t in range(T):
        for i in range(N):
            if not valid_teeth[i]:
                continue
            x_win = pm[t:t + time_win, i:i + tooth_win, :]
            y_win = pe[t:t + time_win, i:i + tooth_win, :]
            full, cs = vector_ssim_window(x_win, y_win, c1, c2)
            full_map[t, i] = full
            cs_map[t, i] = cs

    return full_map, cs_map


def nanmean_map(m: np.ndarray) -> float:
    vals = m[np.isfinite(m)]
    if vals.size == 0:
        return np.nan
    return float(vals.mean())


def remap_signed_to_01(m: np.ndarray) -> np.ndarray:
    return (m + 1.0) / 2.0


# ============================================================
# Visualization
# ============================================================
def save_vfstksm_map(map_2d: np.ndarray,
                     out_path: str,
                     title: str,
                     cfg: dict,
                     signed: bool = True):
    ensure_dir(os.path.dirname(out_path))

    vis = remap_signed_to_01(map_2d) if signed else map_2d.copy()

    plt.figure(figsize=(11, 4.8))
    im = plt.imshow(
        vis.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cfg.get("VIS_CMAP", "viridis"),
        vmin=cfg.get("VIS_VMIN", 0.0),
        vmax=cfg.get("VIS_VMAX", 1.0),
    )
    plt.colorbar(im, fraction=0.026, pad=0.02, label="VF-STKSM similarity")
    plt.xlabel("Normalized treatment progress")
    plt.ylabel("Tooth index")
    plt.yticks(np.arange(len(FDI_IDS)), [str(x) for x in FDI_IDS], fontsize=6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.get("VIS_DPI", 180))
    plt.close()


# ============================================================
# Evaluation
# ============================================================
def evaluate_case(case_id: str, method: dict, expert_traj: dict, cfg: dict):
    method_traj = load_trajectory(method["root"], case_id, method.get("format", "auto"))
    if method_traj is None:
        return None

    valid_teeth = expert_traj["valid_mask"] & method_traj["valid_mask"]
    time_win = cfg["TIME_WIN"]
    tooth_win = cfg["TOOTH_WIN"]

    # Translation VF-STKSM
    vt_e = normalize_time_vector(compute_translation_vectors(expert_traj), cfg["NORM_STEPS"])
    vt_m = normalize_time_vector(compute_translation_vectors(method_traj), cfg["NORM_STEPS"])
    c1_t, c2_t = ssim_constants(cfg, "trans")
    trans_full_map, trans_cs_map = vector_ssim_maps(vt_m, vt_e, valid_teeth, time_win, tooth_win, c1_t, c2_t)

    # Rotation VF-STKSM
    vr_e = normalize_time_vector(compute_rotation_vectors(expert_traj), cfg["NORM_STEPS"])
    vr_m = normalize_time_vector(compute_rotation_vectors(method_traj), cfg["NORM_STEPS"])
    c1_r, c2_r = ssim_constants(cfg, "rot")
    rot_full_map, rot_cs_map = vector_ssim_maps(vr_m, vr_e, valid_teeth, time_win, tooth_win, c1_r, c2_r)

    row = {
        "method": method["name"],
        "case_id": case_id,
        "expert_steps": len(expert_traj["files"]),
        "method_steps": len(method_traj["files"]),
        "valid_teeth": int(valid_teeth.sum()),
        "trans_vssim": nanmean_map(trans_full_map),
        "trans_vssim_cs": nanmean_map(trans_cs_map),
        "rot_vssim": nanmean_map(rot_full_map),
        "rot_vssim_cs": nanmean_map(rot_cs_map),
    }

    maps = {
        "trans_vssim": trans_full_map,
        "trans_vssim_cs": trans_cs_map,
        "rot_vssim": rot_full_map,
        "rot_vssim_cs": rot_cs_map,
    }
    return row, maps


def get_cases_for_method(method: dict, expert_root: str) -> List[str]:
    if not os.path.isdir(method["root"]) or not os.path.isdir(expert_root):
        return []
    cases = sorted(list(set(os.listdir(method["root"])) & set(os.listdir(expert_root))))
    return [c for c in cases if c.startswith("C")]


def finite_mean_std(values):
    arr = np.array([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def write_per_case_csv(rows, out_path):
    fields = [
        "method", "case_id", "expert_steps", "method_steps", "valid_teeth",
        "trans_vssim", "trans_vssim_cs",
        "rot_vssim", "rot_vssim_cs",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_summary_csv(rows, out_path):
    methods = sorted(set(r["method"] for r in rows))
    metrics = ["trans_vssim", "trans_vssim_cs", "rot_vssim", "rot_vssim_cs"]

    summary = []
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        row = {"method": m, "num_cases": len(sub)}
        for metric in metrics:
            mean, std, n = finite_mean_std([r.get(metric, np.nan) for r in sub])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_valid_cases"] = n
        summary.append(row)

    with open(out_path, "w", newline="") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    return summary


def print_summary(summary):
    print("\n" + "=" * 120)
    print("Vector-valued SSIM / VF-STKSM Summary")
    print("=" * 120)
    print(f"{'Method':<18} {'Trans-VSSIM ↑':>22} {'Trans-VSSIM-CS ↑':>22} {'Rot-VSSIM ↑':>22} {'Rot-VSSIM-CS ↑':>22}")
    print("-" * 120)
    for r in summary:
        def fmt(metric):
            return f"{r[metric + '_mean']:.4f} ± {r[metric + '_std']:.4f}"
        print(
            f"{r['method']:<18} "
            f"{fmt('trans_vssim'):>22} "
            f"{fmt('trans_vssim_cs'):>22} "
            f"{fmt('rot_vssim'):>22} "
            f"{fmt('rot_vssim_cs'):>22}"
        )
    print("=" * 120)


def should_save_vis(case_vis_count: Dict[str, int], method_name: str, cfg: dict) -> bool:
    limit = int(cfg.get("VIS_MAX_CASES_PER_METHOD", 8))
    if limit < 0:
        return True
    return case_vis_count.get(method_name, 0) < limit


def mark_saved_vis(case_vis_count: Dict[str, int], method_name: str):
    case_vis_count[method_name] = case_vis_count.get(method_name, 0) + 1


def save_case_visualizations(case_id: str, method_name: str, maps: dict, cfg: dict):
    vis_root = os.path.join(cfg["OUTPUT_DIR"], cfg.get("VIS_DIR_NAME", "vf_stksm_similarity_maps"))
    safe_method = method_name.replace("/", "_").replace(" ", "_")
    out_dir = os.path.join(vis_root, safe_method, case_id)
    ensure_dir(out_dir)

    # Only VF-STKSM Similarity Map, as requested.
    for key, m in maps.items():
        if key == "trans_vssim":
            title = f"{case_id} | {method_name} | Trans VF-STKSM Full VSSIM"
        elif key == "trans_vssim_cs":
            title = f"{case_id} | {method_name} | Trans VF-STKSM VSSIM-CS"
        elif key == "rot_vssim":
            title = f"{case_id} | {method_name} | Rot VF-STKSM Full VSSIM"
        elif key == "rot_vssim_cs":
            title = f"{case_id} | {method_name} | Rot VF-STKSM VSSIM-CS"
        else:
            title = f"{case_id} | {method_name} | {key}"

        out_path = os.path.join(out_dir, f"{key}_map.png")
        save_vfstksm_map(m, out_path, title, cfg, signed=True)


def run_eval(cfg: dict,
             case_id: Optional[str] = None,
             max_cases: Optional[int] = None,
             save_vis_override: Optional[bool] = None):
    ensure_dir(cfg["OUTPUT_DIR"])

    if save_vis_override is not None:
        cfg["SAVE_VIS"] = bool(save_vis_override)

    if case_id:
        cases = [case_id]
    else:
        all_cases = set()
        for m in cfg["METHODS"]:
            all_cases.update(get_cases_for_method(m, cfg["EXPERT_ROOT"]))
        cases = sorted(all_cases)
        mc = cfg.get("MAX_CASES", -1) if max_cases is None else max_cases
        if mc and mc > 0:
            cases = cases[:mc]

    c1_t, c2_t = ssim_constants(cfg, "trans")
    c1_r, c2_r = ssim_constants(cfg, "rot")

    print("=" * 100)
    print("Vector-valued SSIM / VF-STKSM Evaluation")
    print("=" * 100)
    print(f"Expert root:      {cfg['EXPERT_ROOT']}")
    print(f"Output dir:       {cfg['OUTPUT_DIR']}")
    print(f"Norm steps:       {cfg['NORM_STEPS']}")
    print(f"Window:           time={cfg['TIME_WIN']}, tooth={cfg['TOOTH_WIN']}, vector_dim=3")
    print(f"Trans constants:  C1={c1_t:.8g}, C2={c2_t:.8g}")
    print(f"Rot constants:    C1={c1_r:.8g}, C2={c2_r:.8g}")
    print(f"Save vis:         {cfg['SAVE_VIS']}")
    print(f"Vis cases/method: {cfg['VIS_MAX_CASES_PER_METHOD']}")
    print(f"Cases:            {len(cases)}")
    print(f"Methods:          {[m['name'] for m in cfg['METHODS']]}")
    print("=" * 100)

    rows, skipped = [], []
    vis_counts = {}

    for c in tqdm(cases, desc="Cases"):
        expert = load_trajectory(cfg["EXPERT_ROOT"], c, "json")
        if expert is None:
            skipped.append((c, "Expert", "missing or too few expert steps"))
            continue

        for method in cfg["METHODS"]:
            try:
                result = evaluate_case(c, method, expert, cfg)
                if result is None:
                    skipped.append((c, method["name"], "missing or too few method steps"))
                    continue

                row, maps = result
                rows.append(row)

                if cfg.get("SAVE_VIS", True) and should_save_vis(vis_counts, method["name"], cfg):
                    save_case_visualizations(c, method["name"], maps, cfg)
                    mark_saved_vis(vis_counts, method["name"])

            except Exception as e:
                skipped.append((c, method["name"], str(e)))

    if not rows:
        print("\nNo valid rows were produced. First skipped examples:")
        for item in skipped[:20]:
            print("  ", item)
        raise RuntimeError("No valid method/case results found. Please check paths, formats, and skipped errors.")

    per_case_csv = os.path.join(cfg["OUTPUT_DIR"], "vector_ssim_vfstksm_per_case.csv")
    summary_csv = os.path.join(cfg["OUTPUT_DIR"], "vector_ssim_vfstksm_summary.csv")
    summary_txt = os.path.join(cfg["OUTPUT_DIR"], "vector_ssim_vfstksm_summary.txt")

    write_per_case_csv(rows, per_case_csv)
    summary = write_summary_csv(rows, summary_csv)
    print_summary(summary)

    with open(summary_txt, "w") as f:
        f.write("Vector-valued SSIM / VF-STKSM Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Expert root: {cfg['EXPERT_ROOT']}\n")
        f.write(f"Output dir: {cfg['OUTPUT_DIR']}\n")
        f.write(f"Norm steps: {cfg['NORM_STEPS']}\n")
        f.write(f"Window: time={cfg['TIME_WIN']}, tooth={cfg['TOOTH_WIN']}, vector_dim=3\n")
        f.write(f"Trans constants: C1={c1_t:.8g}, C2={c2_t:.8g}\n")
        f.write(f"Rot constants: C1={c1_r:.8g}, C2={c2_r:.8g}\n")
        f.write(f"Valid rows: {len(rows)}\n")
        f.write(f"Skipped rows: {len(skipped)}\n\n")
        f.write("Metrics:\n")
        f.write("  trans_vssim:     full vector-valued SSIM on translation vector field\n")
        f.write("  trans_vssim_cs:  structure-only vector-valued SSIM on translation vector field\n")
        f.write("  rot_vssim:       full vector-valued SSIM on rotation-vector field\n")
        f.write("  rot_vssim_cs:    structure-only vector-valued SSIM on rotation-vector field\n\n")
        f.write("Method summary CSV contains all mean/std metrics.\n")

        if skipped:
            f.write("\nSkipped examples:\n")
            for c, m, reason in skipped[:200]:
                f.write(f"  {c} | {m}: {reason}\n")
            if len(skipped) > 200:
                f.write(f"  ... and {len(skipped) - 200} more\n")

    print(f"\nSaved per-case CSV: {per_case_csv}")
    print(f"Saved summary CSV:  {summary_csv}")
    print(f"Saved summary TXT:  {summary_txt}")
    if cfg.get("SAVE_VIS", True):
        print(f"Saved visualizations under: {os.path.join(cfg['OUTPUT_DIR'], cfg.get('VIS_DIR_NAME', 'vf_stksm_similarity_maps'))}")

    if skipped:
        print(f"Skipped rows: {len(skipped)}")
        for item in skipped[:10]:
            print("  ", item)


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--expert_root", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--case_id", type=str, default=None)
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--norm_steps", type=int, default=None)

    p.add_argument("--time_win", type=int, default=None)
    p.add_argument("--tooth_win", type=int, default=None)

    p.add_argument("--trans_range", type=float, default=None)
    p.add_argument("--rot_range", type=float, default=None)
    p.add_argument("--k1", type=float, default=None)
    p.add_argument("--k2", type=float, default=None)

    p.add_argument("--c1_trans", type=float, default=None)
    p.add_argument("--c2_trans", type=float, default=None)
    p.add_argument("--c1_rot", type=float, default=None)
    p.add_argument("--c2_rot", type=float, default=None)

    p.add_argument("--no_vis", action="store_true")
    p.add_argument("--vis_all", action="store_true", help="Save visualization for all cases/methods.")
    p.add_argument("--vis_max_cases_per_method", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)

    if args.expert_root:
        cfg["EXPERT_ROOT"] = args.expert_root
    if args.output_dir:
        cfg["OUTPUT_DIR"] = args.output_dir
    if args.norm_steps is not None:
        cfg["NORM_STEPS"] = args.norm_steps

    if args.time_win is not None:
        cfg["TIME_WIN"] = args.time_win
    if args.tooth_win is not None:
        cfg["TOOTH_WIN"] = args.tooth_win

    if args.trans_range is not None:
        cfg["TRANS_RANGE_MM"] = args.trans_range
    if args.rot_range is not None:
        cfg["ROT_RANGE_DEG"] = args.rot_range
    if args.k1 is not None:
        cfg["K1"] = args.k1
    if args.k2 is not None:
        cfg["K2"] = args.k2

    if args.c1_trans is not None:
        cfg["C1_TRANS"] = args.c1_trans
    if args.c2_trans is not None:
        cfg["C2_TRANS"] = args.c2_trans
    if args.c1_rot is not None:
        cfg["C1_ROT"] = args.c1_rot
    if args.c2_rot is not None:
        cfg["C2_ROT"] = args.c2_rot

    if args.no_vis:
        cfg["SAVE_VIS"] = False
    if args.vis_all:
        cfg["VIS_MAX_CASES_PER_METHOD"] = -1
    if args.vis_max_cases_per_method is not None:
        cfg["VIS_MAX_CASES_PER_METHOD"] = args.vis_max_cases_per_method

    run_eval(
        cfg,
        case_id=args.case_id,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
