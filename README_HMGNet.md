# HMG-Net: Motion-Gated Autoregressive Learning for 3D Orthodontic Trajectory Generation

This repository contains the implementation of **HMG-Net**, a tooth-wise motion-gated autoregressive model for 3D orthodontic trajectory generation. The framework predicts staged 6-DoF tooth movement trajectories from an initial malocclusion setup to a target occlusion setup.

The model decomposes orthodontic motion generation into two complementary branches:

- **Magnitude path**: predicts continuous translational and rotational motion increments.
- **Activation path**: predicts whether each tooth should move at a given step through translation and rotation gates.

The final action is composed by combining the predicted motion magnitude with the soft gate probability and a hard activation mask.

---

## 1. Project Structure

A recommended project structure is:

```text
HMG-Net/
├── configs/
│   └── ab_06_infer.yaml
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
│       │   └── pointnet.py
│       ├── strategy/
│       │   ├── base.py
│       │   └── dual_gate.py
│       └── head/
│           ├── base.py
│           └── uncertain.py
├── scripts/
│   ├── preprocess.py
│   ├── estimate_motion_label_thresholds_quat_to_6d_config.py
│   └── analyze_tooth_type_staging.py
├── main_train.py
├── infer.py
├── eval.py
└── README.md
```

If the uploaded files have names such as `model(8).py` or `dataset(10).py`, they should be renamed to their clean repository names before running the code.

---

## 2. Environment

The model is implemented in **PyTorch**. The reported experiments were trained on an **NVIDIA GeForce RTX 4090 GPU**.

Recommended dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib tqdm pyyaml tensorboard open3d
```

For collision evaluation, the evaluation script also requires FCL bindings. Depending on your environment, install either:

```bash
pip install python-fcl
```

or install FCL through conda/system packages if `python-fcl` cannot be built directly.

---

## 3. Dataset Format

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

Each step file stores one tooth per line:

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

The preprocessing pipeline converts quaternion rotations into the continuous 6D rotation representation and stores each pose as a 9D vector:

```text
[x, y, z, r1, r2, r3, r4, r5, r6]
```

---

## 4. Preprocessing

Run:

```bash
python scripts/preprocess.py
```

The preprocessing script performs the following operations:

1. Reads all expert step files.
2. Converts quaternion rotations into 6D rotation representations.
3. Constructs a fixed 32-slot tooth representation.
4. Builds a binary valid-tooth mask for missing or extracted teeth.
5. Samples each valid tooth mesh into point clouds for shape encoding.
6. Saves processed tensors for training and inference.

Expected processed files for each case include:

```text
processed_root/
├── case_001/
│   ├── poses_9d.pt
│   ├── shape_feature.pt
│   └── meta.pt
```

---

## 5. Motion-Label Threshold Estimation

The binary movement labels for the activation gates are generated from expert step-wise motion increments. To estimate the motion threshold statistically, run:

```bash
python scripts/estimate_motion_label_thresholds_quat_to_6d_config.py
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
   - Two-component Gaussian mixture modeling

The final experiments adopt the Otsu-based threshold estimates as the statistical basis for activation-label generation.

Example output:

```text
Recommended label-generation thresholds if adopting Otsu:
  translation_mm: 0.093212
  rotation_6d_norm: 0.014229
```

In the final hysteresis-style labeling scheme, the Otsu threshold provides the approximate movement separation point. A higher start threshold and a lower keep threshold are then used to reduce label flickering around near-zero motion.

---

## 6. Tooth-Type Staging Analysis

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

## 7. Training

Training is launched through:

```bash
python main_train.py --config configs/ab_06_infer.yaml
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
    MSE warm-up for stable initialization.

Stage 2: epochs 61-120
    Introduces translational uncertainty learning through Gaussian NLL.

Stage 3 rotation-freeze phase:
    80 epochs with constrained rotation log-variance.

Final joint stage:
    Continues until epoch 400 with full joint optimization.
```

---

## 8. Model Architecture

### 8.1 Shape Encoder

Each tooth mesh is uniformly sampled into 1024 points. A MiniPointNet-style encoder extracts a tooth-level shape embedding:

```text
[B, 32, 1024, 3] -> [B, 32, D]
```

where `D = 512`.

### 8.2 Magnitude Path

For each tooth, the translation state is:

```text
s_T = cat(current position relative to centroid, target-position residual)
```

The rotation state is:

```text
s_R = cat(current 6D rotation, target-rotation residual)
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

### 8.3 Activation Path

The activation path predicts whether each tooth should move at the current step. It uses compact 10D strategy vectors for translation and rotation separately:

```text
tooth type code
category-wise completion vector
residual magnitude
previous activation cue
```

An AdvancedDualStreamMaskHead separates static tooth-type information from dynamic treatment-state information and predicts gate logits for translation and rotation.

### 8.4 Motion-Gated Action Composition

During inference, gate logits are converted to probabilities with a sigmoid function and then thresholded into hard masks. The final update is:

```text
final translation = translation mean × soft translation gate × hard translation mask
final rotation = rotation mean × soft rotation gate × hard rotation mask
```

Uncertainty-based motion scaling is disabled in the reported experiments.

---

## 9. Inference

Run:

```bash
python infer.py
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

## 10. Evaluation

Run:

```bash
python eval.py
```

The evaluation script measures orthodontic trajectories from the following perspectives:

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

## 11. Reproducibility Notes

The reported experiments use:

```text
Dataset size: 8,000 cases
Train/test split: 7,000 / 1,000
Missing-tooth cases among all available cases: 2,693
GPU: NVIDIA GeForce RTX 4090
Training time: approximately 2.5 hours
Inference time: approximately 3 minutes for 1,000 test cases
```

The final checkpoint is selected according to the lowest training objective.

---

## 12. Important Notes

1. The training data are curated to remove trajectories with abnormal stepwise movements exceeding biological movement thresholds.
2. This curation is applied before training and is not used as inference-time clipping.
3. During inference, no biological-threshold-based post-hoc safety clipping is applied.
4. The reported zero-violation result should be interpreted in the context of the curated expert training distribution.
5. The expert trajectories are clinically approved digital treatment plans, not direct in-vivo tooth-motion measurements.
6. Demographic annotations such as age, malocclusion class, and extraction ratio are unavailable due to data-provider privacy constraints.

---

## 13. Common Issues

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
