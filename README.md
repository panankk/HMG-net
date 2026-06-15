# HMG-Net: Motion-Gated Autoregressive Learning for 3D Orthodontic Trajectory Generation

This repository contains the implementation of **HMG-Net**, a motion-gated autoregressive framework for 3D orthodontic trajectory generation. HMG-Net predicts staged 6-DoF tooth movement trajectories from an initial malocclusion setup to a target occlusion setup.

The model decomposes orthodontic trajectory generation into two complementary branches:

- **Magnitude path**: predicts continuous translational and rotational motion increments.
- **Activation path**: predicts whether each tooth should move at a given step through translation and rotation gates.

The final action is composed by combining the predicted motion magnitude with the soft gate probability and a hard activation mask.

> **Data availability.** Due to clinical data-sharing and privacy restrictions, the real-world orthodontic dataset used in the paper cannot be publicly released. This repository provides the full model implementation, preprocessing interface, training/inference pipeline, evaluation scripts, analysis tools, and expected data format.

---

## 1. Repository Structure

The code is organized as follows:

```text
HMG-net/
├── configs/
│   ├── train_full.yaml
│   ├── infer_full.yaml
│   └── ablations/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── engine/
│   ├── __init__.py
│   ├── losses.py
│   └── trainer.py
├── models/
│   ├── __init__.py
│   ├── model.py
│   └── components/
│       ├── backbone/
│       │   ├── base.py
│       │   ├── identity.py
│       │   └── pointnet.py
│       ├── strategy/
│       │   ├── base.py
│       │   ├── identity.py
│       │   ├── single_stream.py
│       │   └── dual_gate.py
│       └── head/
│           ├── base.py
│           ├── deterministic.py
│           ├── identity.py
│           └── uncertain.py
├── scripts/
│   ├── preprocess.py
│   ├── estimate_motion_label_thresholds.py
│   ├── analyze_tooth_type_staging.py
│   └── evaluate_vssim.py
├── main_train.py
├── infer.py
├── eval.py
├── requirements.txt
├── environment.yml
└── README.md
```

The root directory only contains the main training, inference, and evaluation entry points. Preprocessing and auxiliary analysis scripts are placed under `scripts/`.

---

## 2. Installation

The reported experiments were conducted with **PyTorch** on an **NVIDIA GeForce RTX 4090 GPU**.

### Option A: Install from `requirements.txt`

```bash
conda create -n hmgnet python=3.9 -y
conda activate hmgnet

# Install PyTorch. Please change cu118 according to your CUDA version if needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies.
pip install -r requirements.txt
```

### Option B: Install from `environment.yml`

```bash
conda env create -f environment.yml
conda activate hmgnet
```

`python-fcl` is required only for collision evaluation. If `python-fcl` fails to install through pip, please install FCL through conda or system packages first, and then install the Python binding.

---

## 3. Data Format

The raw expert trajectory data should be organized by case:

```text
raw_data_root/
├── case_001/
│   ├── step0.txt
│   ├── step1.txt
│   └── ...
├── case_002/
│   ├── step0.txt
│   ├── step1.txt
│   └── ...
```

Each step file stores one tooth pose per line:

```text
FDI x y z qx qy qz qw
```

Example:

```text
27 -22.6665 15.3546 -6.0654 0.809348 -0.56267 0.168178 0.0085847
```

where:

- `FDI` is the tooth ID.
- `x y z` are the tooth translation coordinates.
- `qx qy qz qw` is the tooth rotation quaternion.

The preprocessing pipeline converts quaternion rotations into the continuous 6D rotation representation and stores each tooth pose as a 9D vector:

```text
[x, y, z, r1, r2, r3, r4, r5, r6]
```

Missing or extracted teeth are handled by a binary tooth-validity mask.

---

## 4. Configuration

All local paths should be specified in YAML configuration files. Do **not** hard-code personal absolute paths such as `/home/username/...` inside Python scripts.

A typical training configuration should contain:

```yaml
DATA:
  PROCESSED_ROOT: "/path/to/processed_train_data"

MODEL:
  BACKBONE:
    NAME: "MiniPointNet"
  STRATEGY:
    NAME: "AdvancedDualStreamMaskHead"
  HEAD:
    POS:
      NAME: "UncertainRegressionHead"
    ROT:
      NAME: "UncertainRegressionHead"
  D_MODEL: 512
  NHEAD: 8
  NUM_LAYERS: 6
  DROPOUT: 0.1
  NUM_TEETH: 32
  WINDOW_SIZE: 1
  USE_WINDOW_ATTN: false
  USE_SPATIAL_ATTN: true
  USE_TOOTH_TYPE: true
  USE_GLOBAL_TIME: true
```

A typical inference/evaluation configuration should contain:

```yaml
DATA:
  PROCESSED_ROOT: "/path/to/processed_test_data"
  EXPERT_ROOT: "/path/to/raw_test_data"
  HULL_ROOT: "/path/to/tooth_hulls"
  REMOVE_JSON: "/path/to/remove_idx_summary.json"

INFER:
  CKPT_PATH: "/path/to/checkpoint.pth"
  SAVE_ROOT: "/path/to/inference_results"
  MAX_STEPS: 150
  STOP_THRES_POS: 0.2
  STOP_THRES_DEG: 3.0
  MASK_THRESHOLD: 0.5
  UNCERTAINTY_SCALE_FACTOR: 0.0
```

Please replace all `/path/to/...` fields with your local paths.

---

## 5. Preprocessing

Run:

```bash
python scripts/preprocess.py --config configs/train_full.yaml
```

The preprocessing script performs the following operations:

1. Reads all expert step files.
2. Converts quaternion rotations into 6D rotation representations.
3. Constructs a fixed 32-slot tooth representation.
4. Builds a binary valid-tooth mask for missing or extracted teeth.
5. Samples each valid tooth mesh into 1024 points for shape encoding.
6. Optionally generates tooth hulls for collision evaluation.
7. Saves processed tensors for training and inference.

Expected processed files for each case include pose tensors, tooth point-cloud tensors, validity masks, and metadata required by the dataset loader.

---

## 6. Motion-Label Threshold Estimation

The binary movement labels for the activation gates are generated from expert step-wise motion increments. To estimate the movement threshold statistically, run:

```bash
python scripts/estimate_motion_label_thresholds.py
```

This script:

1. Reads expert trajectories.
2. Computes step-wise translation increments:

```text
||T^{t+1} - T^t||_2
```

3. Converts quaternion rotations to 6D rotation representations.
4. Computes step-wise 6D rotation increments:

```text
||R6D^{t+1} - R6D^t||_2
```

5. Performs histogram analysis.
6. Estimates thresholds using:
   - Otsu thresholding
   - two-component Gaussian mixture modeling

The final experiments adopt the Otsu-based threshold estimates as the statistical basis for activation-label generation.

Example output:

```text
Recommended label-generation thresholds if adopting Otsu:
  translation_mm: 0.093212
  rotation_6d_norm: 0.014229
```

In the final hysteresis-style labeling scheme, the Otsu threshold provides the approximate movement separation point. A higher start threshold and a lower keep threshold are used to reduce label flickering around near-zero motion.

> Thresholds used for label generation should be estimated from the **training expert trajectories**, not from the test set.

---

## 7. Tooth-Type Staging Analysis

To analyze whether different tooth types exhibit different stage-wise movement patterns, run:

```bash
python scripts/analyze_tooth_type_staging.py
```

The script divides teeth into four categories:

```text
incisor / canine / premolar / molar
```

It then computes activation probability across normalized treatment phases, such as early, middle, and late stages. This analysis supports the use of category-wise treatment-progress features in the activation path.

---

## 8. Training

Training is launched through:

```bash
python main_train.py --config configs/train_full.yaml
```

The training script:

1. Loads the YAML configuration.
2. Builds the dataset through `build_dataset`.
3. Builds the model through `build_model`.
4. Constructs the multi-task loss.
5. Trains the model with a staged curriculum.

The reported full setting uses:

```text
Hidden dimension: 512
Attention heads: 8
Spatial Transformer layers: 6
Dropout: 0.1
Batch size: 32
Training epochs: 400
Optimizer: AdamW
Initial learning rate: 1e-3
Weight decay: 1e-4
Warm-up epochs: 15
Random seed: 40
Gradient clipping: 1.0
Input window size: W = 1
Temporal-window attention: disabled
Spatial Transformer: enabled
Tooth-type conditioning: enabled
```

Training follows a staged curriculum:

```text
Stage 1: epochs 1-60
    MSE-style warm-up for stable initialization.

Stage 2: epochs 61-120
    Introduces translational uncertainty learning through Gaussian NLL.

Stage 3 rotation-freeze phase:
    80 epochs with constrained rotation log-variance.

Final joint stage:
    Continues until epoch 400 with full joint optimization.
```

The final checkpoint is selected according to the lowest training objective.

---

## 9. Model Architecture

### 9.1 Shape Encoder

Each valid tooth mesh is uniformly sampled into 1024 points. A MiniPointNet-style encoder extracts a tooth-level shape embedding:

```text
[B, 32, 1024, 3] -> [B, 32, D]
```

where `D = 512`.

### 9.2 Magnitude Path

For each tooth, the translation state is:

```text
s_T = concat(current position relative to centroid, target-position residual)
```

The rotation state is:

```text
s_R = concat(current 6D rotation, target-rotation residual)
```

The translation and rotation states are separately embedded and fused with:

- tooth shape embedding
- tooth ID embedding
- tooth type embedding
- sinusoidal time-step embedding

The final tooth tokens are processed by two independent spatial Transformer encoders, one for translation and one for rotation.

The regression heads predict:

```text
translation mean and log-variance
rotation mean and log-variance
```

The log-variance terms are used during training for heteroscedastic likelihood regularization.

### 9.3 Activation Path

The activation path predicts whether each tooth should move at the current step. It uses compact 10D strategy vectors for translation and rotation separately:

```text
tooth type code
category-wise completion vector
residual magnitude
previous activation cue
```

An AdvancedDualStreamMaskHead separates static tooth-type information from dynamic treatment-state information and predicts gate logits for translation and rotation.

### 9.4 Motion-Gated Action Composition

During inference, gate logits are converted to probabilities with a sigmoid function and then thresholded into hard masks. The final update is:

```text
final translation = translation mean × soft translation gate × hard translation mask
final rotation = rotation mean × soft rotation gate × hard rotation mask
```

Uncertainty-based motion scaling is disabled in the reported experiments by setting the uncertainty scale factor to `0.0`.

---

## 10. Inference

Run:

```bash
python infer.py --config configs/infer_full.yaml
```

The inference pipeline:

1. Loads the trained checkpoint.
2. Loads each processed test case.
3. Starts from the initial pose.
4. Autoregressively predicts step-wise tooth updates.
5. Stops when convergence criteria are met or when the maximum rollout horizon is reached.
6. Saves each generated step as a text file.

Reported inference configuration:

```text
Maximum rollout horizon: 150
Translation stopping threshold: 0.2 mm
Rotation stopping threshold: 3 degrees
Gate threshold: 0.5
Uncertainty scale factor: 0.0
```

---

## 11. Evaluation

Run:

```bash
python eval.py --config configs/infer_full.yaml
```

The evaluation script measures orthodontic trajectories from the following perspectives.

### Movement efficiency

```text
sum_T: cumulative translation
sum_R: cumulative rotation
Average Steps: generated treatment length
Stage Difference: difference from expert treatment length
```

### Step-wise safety

A violation is counted when:

```text
translation > 0.5 mm
rotation > 3 degrees
```

Negligible translational jitter below 0.01 mm is ignored.

### Collision frequency

Mesh-level inter-tooth collision is detected with FCL on adjacent tooth pairs. A collision event is counted when the mean penetration depth exceeds 0.3 mm.

The normalized collision frequency is:

```text
f_coll = N_coll / (N_case × 28)
```

This is a trajectory-level collision frequency and is not strictly bounded by 1.

### Target-reaching success

A case is considered successful if it converges to the target occlusion under the inference stopping criteria.

---

## 12. STKSM / VSSIM Evaluation

To evaluate spatio-temporal kinematic similarity, run:

```bash
python scripts/evaluate_vssim.py
```

This script compares predicted and expert trajectories using vector-valued structural similarity over tooth-time kinematic fields. Translation and rotation motion fields are temporally normalized and compared over the time-tooth plane.

---

## 13. Reproducibility Notes

The reported experiments use:

```text
Dataset size: 8,000 cases
Train/test split: 7,000 / 1,000
Missing-tooth cases among all available cases: 2,693
GPU: NVIDIA GeForce RTX 4090
Training time: approximately 2.5 hours
Inference time: approximately 3 minutes for 1,000 test cases
```

The expert trajectories are clinically approved digital treatment plans rather than direct in-vivo tooth motion measurements. Detailed demographic annotations, such as patient age, malocclusion type, and extraction ratio, are unavailable due to provider privacy constraints.

---

## 14. Important Notes

1. The training data are curated to remove trajectories with abnormal stepwise movements exceeding biological movement thresholds.
2. This curation is applied before training and is not used as inference-time clipping.
3. During inference, no biological-threshold-based post-hoc safety clipping is applied.
4. The reported zero-violation result should be interpreted in the context of the curated expert training distribution.
5. Clinical trajectory data are not released due to privacy restrictions.
6. All local paths should be specified in YAML configuration files.

---

## 15. Common Issues

### Python imports fail

Make sure the repository uses the expected package structure:

```text
data/
engine/
models/
models/components/
```

and each package contains an `__init__.py` file.

### FCL installation fails

Collision evaluation depends on FCL. If `pip install python-fcl` fails, install system-level FCL or use a conda environment.

### Rotation threshold mismatch

The model uses 6D rotation representation internally. Therefore, rotation-label thresholds should be estimated using the 6D rotation difference norm, not quaternion angular degrees.

### Test data should not be used to select thresholds

Thresholds used for label generation should be estimated from the training expert trajectories. Test data should only be used for final evaluation.
