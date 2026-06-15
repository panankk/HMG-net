#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
estimate_motion_label_thresholds_quat_to_6d_config.py

直接运行版：从专家 step 文件中读取四元数 [qx,qy,qz,qw]，
先转换为 rotation matrix，再取前两列组成 6D rotation representation，
最后统计相邻 step 的 6D rotation difference norm 阈值。

适用于你给出的这种单步格式：
FDI x y z qx qy qz qw

例如：
27 -22.6665 15.3546 -6.0654 0.809348 -0.56267 0.168178 0.0085847

功能：
1. 遍历 expert_root 下所有病例。
2. 支持每个 step 是 .txt 或 .json。
3. 计算相邻 step 的：
   - translation increment: ||T^{t+1} - T^t||_2, 单位 mm
   - rotation_6d increment: ||R6D^{t+1} - R6D^t||_2
4. 对两种 increment 分别做直方图统计。
5. 用 Otsu 大津算法和 2-component GMM 估计阈值。
6. 输出 threshold_summary.csv 和两张直方图。

运行：
python estimate_motion_label_thresholds_quat_to_6d_config.py
或：
python3 estimate_motion_label_thresholds_quat_to_6d_config.py
"""

import os
import re
import csv
import json
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 直接在这里改参数
# ============================================================
CONFIG = {
    # 专家轨迹根目录：里面应该是一堆病例文件夹，每个病例下有 step_*.txt 或 step_*.json
    "expert_root": "/home/pa/version4/data/test_data",

    # 输出目录
    "output_dir": "threshold_analysis_quat_to_6d",

    # 分析多少病例；-1 表示全部
    "num_cases": 1000,

    # 随机种子
    "seed": 42,

    # step 文件类型：
    # "auto": 自动读取 json 和 txt
    # "txt": 只读取 txt
    # "json": 只读取 json
    "file_mode": "auto",

    # 直方图 bins 数
    "bins": 512,

    # Otsu 和画图用的上限分位数，避免极端 outlier 拉长横轴
    "hist_percentile": 99.5,

    # GMM 最多采样点数，避免太慢
    "gmm_max_samples": 300000,

    # 四元数顺序。你给的样例是 qx qy qz qw，所以保持 xyzw。
    # 可选："xyzw" 或 "wxyz"
    "quat_order": "xyzw",

    # FDI 牙位顺序
    "fdi_ids": [
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        48, 47, 46, 45, 44, 43, 42, 41,
        31, 32, 33, 34, 35, 36, 37, 38
    ],
}
# ============================================================


def parse_step(filename):
    base = os.path.basename(filename)
    match = re.search(r"step_?(\d+)", base)
    if match:
        return int(match.group(1))
    return -1


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return None
    return q / norm


def quat_xyzw_to_rotmat(q):
    """
    q = [qx, qy, qz, qw]
    return 3x3 rotation matrix.
    """
    q = normalize_quat(q)
    if q is None:
        return None

    x, y, z, w = q

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    R = np.array([
        [ww + xx - yy - zz, 2.0 * (xy - zw),     2.0 * (xz + yw)],
        [2.0 * (xy + zw),     ww - xx + yy - zz, 2.0 * (yz - xw)],
        [2.0 * (xz - yw),     2.0 * (yz + xw),     ww - xx - yy + zz],
    ], dtype=np.float64)

    return R


def quat_to_6d(q, quat_order):
    """
    Convert quaternion to 6D rotation representation.
    6D = first two columns of rotation matrix, flattened as [col1, col2].
    """
    q = np.asarray(q, dtype=np.float64)

    if quat_order == "wxyz":
        # input [qw,qx,qy,qz] -> [qx,qy,qz,qw]
        q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)

    R = quat_xyzw_to_rotmat(q)
    if R is None:
        return None

    # Zhou 6D representation usually uses first two columns.
    # Flatten column-wise: [R00,R10,R20,R01,R11,R21]
    rot6d = R[:, :2].reshape(6, order="F")
    return rot6d


def read_txt_step(path, fdi_ids, quat_order):
    """
    读取 txt step 文件。
    每行格式：
    FDI x y z qx qy qz qw
    """
    positions = {}
    rot6ds = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                fdi = int(parts[0])
            except Exception:
                continue

            if fdi not in fdi_ids:
                continue

            try:
                vals = [float(x) for x in parts[1:8]]
            except Exception:
                continue

            pos = np.asarray(vals[:3], dtype=np.float64)
            quat = np.asarray(vals[3:7], dtype=np.float64)
            r6 = quat_to_6d(quat, quat_order)
            if r6 is None:
                continue

            positions[fdi] = pos
            rot6ds[fdi] = r6

    return positions, rot6ds


def read_json_step(path, fdi_ids, quat_order):
    """
    读取 json step 文件。
    默认每个 key 是 FDI，value 至少为 [x,y,z,qx,qy,qz,qw]。
    """
    positions = {}
    rot6ds = {}

    with open(path, "r") as f:
        data = json.load(f)

    for fdi_str, vals in data.items():
        try:
            fdi = int(fdi_str)
        except Exception:
            continue

        if fdi not in fdi_ids:
            continue

        vals = list(vals)
        if len(vals) < 7:
            continue

        try:
            arr = np.asarray(vals[:7], dtype=np.float64)
        except Exception:
            continue

        pos = arr[:3]
        quat = arr[3:7]
        r6 = quat_to_6d(quat, quat_order)
        if r6 is None:
            continue

        positions[fdi] = pos
        rot6ds[fdi] = r6

    return positions, rot6ds


def find_step_files(case_dir, file_mode):
    files = []

    if file_mode in ("auto", "json"):
        files.extend(glob.glob(os.path.join(case_dir, "*step*.json")))

    if file_mode in ("auto", "txt"):
        files.extend(glob.glob(os.path.join(case_dir, "*step*.txt")))

    files = sorted(files, key=parse_step)
    return files


def read_step(path, fdi_ids, quat_order):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt_step(path, fdi_ids, quat_order)
    elif ext == ".json":
        return read_json_step(path, fdi_ids, quat_order)
    else:
        return {}, {}


def collect_increments(config):
    expert_root = config["expert_root"]
    file_mode = config["file_mode"]
    fdi_ids = config["fdi_ids"]
    quat_order = config["quat_order"]

    case_ids = sorted([
        d for d in os.listdir(expert_root)
        if os.path.isdir(os.path.join(expert_root, d))
    ])

    if config["num_cases"] > 0 and config["num_cases"] < len(case_ids):
        random.seed(config["seed"])
        case_ids = random.sample(case_ids, config["num_cases"])

    trans_increments = []
    rot6d_increments = []

    print("Analyzing {} cases from: {}".format(len(case_ids), expert_root))

    for case_id in tqdm(case_ids):
        case_dir = os.path.join(expert_root, case_id)
        files = find_step_files(case_dir, file_mode)

        if len(files) < 2:
            continue

        # 逐步读取，避免一次性占太多内存
        prev_pos, prev_r6 = read_step(files[0], fdi_ids, quat_order)

        for t in range(1, len(files)):
            curr_pos, curr_r6 = read_step(files[t], fdi_ids, quat_order)

            common_fdis = sorted(set(prev_pos.keys()) & set(curr_pos.keys()) & set(prev_r6.keys()) & set(curr_r6.keys()))

            for fdi in common_fdis:
                d_pos = float(np.linalg.norm(curr_pos[fdi] - prev_pos[fdi]))
                d_r6 = float(np.linalg.norm(curr_r6[fdi] - prev_r6[fdi]))

                if np.isfinite(d_pos):
                    trans_increments.append(d_pos)
                if np.isfinite(d_r6):
                    rot6d_increments.append(d_r6)

            prev_pos, prev_r6 = curr_pos, curr_r6

    return np.asarray(trans_increments, dtype=np.float64), np.asarray(rot6d_increments, dtype=np.float64)


def otsu_threshold(values, bins, value_range=None):
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if len(values) == 0:
        return float("nan")

    if value_range is None:
        vmax = float(np.percentile(values, 99.5))
        if vmax <= 0:
            vmax = float(np.max(values))
        value_range = (0.0, vmax)

    hist, edges = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = hist.sum()
    if total <= 0:
        return float("nan")

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    sigma_b2 = np.zeros_like(centers)
    valid = denom > 1e-12
    sigma_b2[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denom[valid]

    best_idx = int(np.argmax(sigma_b2))
    return float(centers[best_idx])


def gmm_threshold(values, max_samples):
    result = {
        "threshold": float("nan"),
        "mean_low": float("nan"),
        "mean_high": float("nan"),
        "std_low": float("nan"),
        "std_high": float("nan"),
        "weight_low": float("nan"),
        "weight_high": float("nan"),
    }

    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if len(values) < 10:
        return result

    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        print("Warning: sklearn is not available. Skipping GMM.")
        return result

    rng = np.random.RandomState(42)
    if len(values) > max_samples:
        idx = rng.choice(len(values), size=max_samples, replace=False)
        values_fit = values[idx]
    else:
        values_fit = values

    X = values_fit.reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=42,
            n_init=5,
            max_iter=500,
        )
        gmm.fit(X)
    except Exception as e:
        print("Warning: GMM fitting failed: {}".format(e))
        return result

    means = gmm.means_.flatten()
    vars_ = gmm.covariances_.reshape(2)
    stds = np.sqrt(np.maximum(vars_, 1e-12))
    weights = gmm.weights_.flatten()

    order = np.argsort(means)
    m1 = means[order[0]]
    m2 = means[order[1]]
    s1 = stds[order[0]]
    s2 = stds[order[1]]
    w1 = weights[order[0]]
    w2 = weights[order[1]]

    A = 1.0 / (2.0 * s2 ** 2) - 1.0 / (2.0 * s1 ** 2)
    B = m1 / (s1 ** 2) - m2 / (s2 ** 2)
    C = (m2 ** 2) / (2.0 * s2 ** 2) - (m1 ** 2) / (2.0 * s1 ** 2) + np.log((w1 * s2) / (w2 * s1))

    candidates = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            candidates = [-C / B]
    else:
        disc = B ** 2 - 4.0 * A * C
        if disc >= 0:
            candidates = [
                (-B + np.sqrt(disc)) / (2.0 * A),
                (-B - np.sqrt(disc)) / (2.0 * A),
            ]

    threshold = float("nan")
    between = [x for x in candidates if m1 <= x <= m2]
    if len(between) > 0:
        threshold = float(between[0])
    elif len(candidates) > 0:
        mid = 0.5 * (m1 + m2)
        threshold = float(min(candidates, key=lambda x: abs(x - mid)))

    result["threshold"] = threshold
    result["mean_low"] = float(m1)
    result["mean_high"] = float(m2)
    result["std_low"] = float(s1)
    result["std_high"] = float(s2)
    result["weight_low"] = float(w1)
    result["weight_high"] = float(w2)
    return result


def summarize(values):
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    out = {
        "n": float(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "std": float(np.std(values)) if len(values) else float("nan"),
        "min": float(np.min(values)) if len(values) else float("nan"),
        "max": float(np.max(values)) if len(values) else float("nan"),
    }

    for p in [50, 75, 90, 95, 97.5, 99, 99.5, 99.9]:
        out["p{}".format(p)] = float(np.percentile(values, p)) if len(values) else float("nan")

    return out


def plot_histogram(values, otsu_t, gmm_t, title, xlabel, save_path, bins, value_range):
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=bins, range=value_range, alpha=0.75)

    if np.isfinite(otsu_t):
        plt.axvline(otsu_t, linestyle="--", linewidth=2, label="Otsu = {:.6f}".format(otsu_t))
    if np.isfinite(gmm_t):
        plt.axvline(gmm_t, linestyle=":", linewidth=2, label="GMM = {:.6f}".format(gmm_t))

    plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def analyze_metric(values, metric_name, unit_label, output_dir, config):
    values = values[np.isfinite(values)]
    values = values[values >= 0]

    if len(values) == 0:
        print("No valid values for {}".format(metric_name))
        return None

    hist_max = float(np.percentile(values, config["hist_percentile"]))
    if hist_max <= 0:
        hist_max = float(np.max(values))
    value_range = (0.0, hist_max)

    otsu_t = otsu_threshold(values, config["bins"], value_range)
    gmm = gmm_threshold(values, config["gmm_max_samples"])
    summary = summarize(values)

    row = {"metric": metric_name}
    row.update(summary)
    row.update({
        "otsu_threshold": otsu_t,
        "gmm_threshold": gmm["threshold"],
        "gmm_mean_low": gmm["mean_low"],
        "gmm_mean_high": gmm["mean_high"],
        "gmm_std_low": gmm["std_low"],
        "gmm_std_high": gmm["std_high"],
        "gmm_weight_low": gmm["weight_low"],
        "gmm_weight_high": gmm["weight_high"],
    })

    save_path = os.path.join(output_dir, metric_name + "_histogram.png")
    plot_histogram(
        values=values,
        otsu_t=otsu_t,
        gmm_t=gmm["threshold"],
        title="Expert Step-wise {} Increment Distribution".format(metric_name),
        xlabel="{} increment ({})".format(metric_name, unit_label),
        save_path=save_path,
        bins=config["bins"],
        value_range=value_range,
    )

    print("\n{} threshold analysis:".format(metric_name))
    print("  Otsu threshold: {:.6f} {}".format(otsu_t, unit_label))
    print("  GMM threshold:  {:.6f} {}".format(gmm["threshold"], unit_label))
    print("  GMM means:      {:.6f}, {:.6f}".format(gmm["mean_low"], gmm["mean_high"]))
    print("  Histogram saved: {}".format(save_path))

    return row


def write_summary_csv(path, rows):
    fieldnames = [
        "metric",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "p50",
        "p75",
        "p90",
        "p95",
        "p97.5",
        "p99",
        "p99.5",
        "p99.9",
        "otsu_threshold",
        "gmm_threshold",
        "gmm_mean_low",
        "gmm_mean_high",
        "gmm_std_low",
        "gmm_std_high",
        "gmm_weight_low",
        "gmm_weight_high",
    ]

    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    config = CONFIG

    if not os.path.exists(config["expert_root"]):
        raise RuntimeError("expert_root does not exist: {}".format(config["expert_root"]))

    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])

    print("=" * 70)
    print("Expert Motion Threshold Estimation: Quaternion to 6D Rotation Norm")
    print("=" * 70)
    print("expert_root: {}".format(config["expert_root"]))
    print("output_dir: {}".format(config["output_dir"]))
    print("file_mode: {}".format(config["file_mode"]))
    print("quat_order: {}".format(config["quat_order"]))
    print("num_cases: {}".format(config["num_cases"]))
    print("=" * 70)

    trans, rot6d = collect_increments(config)

    print("\nCollected increments:")
    print("  Translation samples: {}".format(len(trans)))
    print("  Rotation-6D samples: {}".format(len(rot6d)))

    rows = []

    row_t = analyze_metric(
        values=trans,
        metric_name="translation_mm",
        unit_label="mm",
        output_dir=config["output_dir"],
        config=config,
    )
    if row_t is not None:
        rows.append(row_t)

    row_r = analyze_metric(
        values=rot6d,
        metric_name="rotation_6d_norm",
        unit_label="6D norm",
        output_dir=config["output_dir"],
        config=config,
    )
    if row_r is not None:
        rows.append(row_r)

    summary_path = os.path.join(config["output_dir"], "threshold_summary.csv")
    write_summary_csv(summary_path, rows)

    print("\nDone.")
    print("Saved summary CSV: {}".format(summary_path))
    print("\nRecommended label-generation thresholds if adopting Otsu:")
    for row in rows:
        print("  {}: {:.6f}".format(row["metric"], float(row["otsu_threshold"])))


if __name__ == "__main__":
    main()
