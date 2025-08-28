<div align="center">

# Real-Time Automatic Detection of Nutrient Deficiency in Lettuce Plants With New YOLOV12 model

Reproducible experiments and comparative training outputs for the paper:
"Real‐Time Automatic Detection of Nutrient Deficiency in Lettuce Plants With New YOLOV12 model"
</div>

## 📌 Paper-Based Summary
This repository accompanies the paper "Real‐Time Automatic Detection of Nutrient Deficiency in Lettuce Plants With New YOLOV12 model" (Journal of Sensors, 2025). The study targets fast and accurate recognition of macro/micro nutrient deficiencies (Calcium, Magnesium, Nitrogen, Phosphorus, Potassium) in hydroponically grown lettuce to enable timely intervention, improved input efficiency, and sustainable production. Traditional visual scouting is slow, subjective, and late‑stage; the paper demonstrates that modern YOLO architectures (including two proposed variants YOLOV11 and YOLOV12 with novel attention and structural modules) enable real‑time detection with high mAP while remaining deployable on constrained hardware.

Key contributions (summarized from the paper):
- Original hydroponic lettuce nutrient deficiency dataset (six classes) captured under real conditions; manually annotated multi‑object images.
- Two new architectures (YOLOV11, YOLOV12) extending recent Ultralytics designs with hierarchical feature attention (HFAM) and dynamic context enhancement (DCEL), plus efficiency blocks (C2PSA, C3k2 revisions) for better small‑object sensitivity and latency.
- Extensive comparison against up‑to‑date YOLO family models under identical hyperparameters (100 epochs, fixed image size, consistent augmentation pipeline) including multi‑scenario data augmentation scaling.
- Empirical evidence that architectural attention + balanced augmentation improves generalization (mAP@50–95) without prohibitive inference cost.

> Note: This is a concise paraphrased summary, not the verbatim abstract. Refer to the PDF for authoritative wording.

## 🎯 Objective
Early and reliable detection of macro nutrient deficiencies (e.g., Nitrogen, Phosphorus, Potassium) on lettuce leaves to reduce yield loss, optimize fertilizer application, and support sustainable cultivation. We benchmark several YOLO family variants (v5, v6, v8, v9) focusing on the accuracy vs. latency trade‑off.

## 📁 Repository Structure
```
├── Journal of Sensors - 2025 - Bakir - Real‐Time Automatic Detection ... .pdf   # Paper
├── yolov5/   # Notebook + training outputs (YOLOv5)
├── yolov6/   # Notebook + training outputs (YOLOv6)
├── yolov8/   # Notebook + training outputs (YOLOv8)
├── yolovv9/  # Notebook + training outputs (YOLOv9, experimental)
└── LICENSE   # Apache 2.0
```
Each model folder contains:
- `Train*.ipynb`: Colab/local training notebook
- `runs/detect/train/args.yaml`: Hyperparameters
- `runs/detect/train/results.csv`: Epoch metrics
- Diagnostic images: confusion matrices, PR curves, label distributions, sample batches

## ⚙️ Training Configuration (Common Core)
Extracted from `args.yaml` files:
- Epochs: 100
- Image size: 640×640
- Augmentation: Mosaic (disabled after epoch 10), RandAugment, random erasing
- Optimizer: `auto` (framework resolves to SGD/AdamW)
- Learning rate: `lr0=0.01`, final factor `lrf=0.01`
- Batch sizes: v5=32, v6=64, v8=40, v9=see notebook/args (same pattern)
- IoU threshold (val): 0.7
- Reproducibility: `seed=0`, `deterministic=true`
Dataset path placeholders use `/content/data.yaml`; you must supply your own YOLO-format `data.yaml`.

### Paper vs Repository Settings
The repository currently includes training artifacts for YOLOv5/6/8/9 using 640×640 images (per `args.yaml`). The paper's experimental section for the proposed YOLOV11 / YOLOV12 models used:
- Image size: 224×224 (to emphasize speed + compactness for real-time hydroponic deployment)
- Epochs: 100 (with early stopping monitored on validation)
- Batch size: 40
- Optimizer: Adam (adaptive per-parameter learning rates)
- Initial learning rate: 0.001 (decay schedule applied as needed)
- Momentum (β term equivalence): 0.937
- IoU threshold (evaluation mAP baseline): 0.5 (reported also as mAP@50 and mAP@50–95 range)
- Augmentation: geometric (rotations incl. 90°), flips, scaling, cropping, brightness/contrast, noise injection, GAN-based synthetic expansion (Scenarios 2–5)

You can reproduce the paper settings (for new architectures) by adjusting Ultralytics CLI:
```bash
yolo detect train model=yolov8m.pt data=data.yaml imgsz=224 epochs=100 batch=40 optimizer=Adam lr0=0.001 project=runs224 name=baseline224
```
Then progressively add augmentation / synthetic data to emulate Scenarios 2–5.

## 🗃 Dataset (From Paper)
Hydroponic lettuce images labeled for six classes:
- Healthy
- Calcium deficiency
- Magnesium deficiency
- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency

Original captured images: 1,086 (real facility). Through preprocessing + augmentation (rotation, flips, noise, brightness/contrast, cropping, scaling, GAN synthesis) total image counts expanded across scenarios (see below) to mitigate class imbalance and reduce overfitting risk.

### Annotation Distribution (excerpt)
Counts below (Train / Validation) reflect label instances, not unique images:
| Class | Train | Val |
|-------|-------|-----|
| Healthy | 2650 | 517 |
| Calcium deficiency | 2655 | 496 |
| Phosphorus deficiency | 2578 | 498 |
| Nitrogen deficiency | 2560 | 498 |
| Potassium deficiency | 2287 | 484 |
| Magnesium deficiency | 1900 | 398 |

### Augmentation Scenarios (as defined in paper)
| Scenario | Total Images | Description (succinct) |
|----------|--------------|-------------------------|
| 1 | 2,978 | Base (original + basic geometric ops) |
| 2 | 4,455 | Scenario 1 + extended rotations/translations |
| 3 | 5,692 | Scenario 2 + contrast/brightness/noise refinements |
| 4 | 7,734 | Scenario 3 + additional scaling/cropping mixes |
| 5 | 10,315 | Scenario 4 + GAN synthetic expansion (balanced classes) |

Data augmentation goals: (a) increase minority class representation, (b) enrich color/texture variance, (c) stress-test small lesion patterns, (d) improve robustness to illumination shifts typical of greenhouse/hydroponic environments.

### Preprocessing Pipeline (Paper Detail)
1. Manual labeling (YOLO format; multiple bounding boxes per image allowed).
2. Resize to 224×224 (paper experiment) OR 640×640 (repo YOLOv5/6/8/9 runs) – choose based on target latency profile.
3. Normalize pixel values to [0,1]; optional standardization if integrating with custom backbones.
4. Apply scenario-specific augmentation set (see table) maintaining class balance ratios.
5. Split: 80% train / 20% validation (paper); maintain stratification across deficiency classes.
6. Optionally curate a held-out test or cross-validation folds for robustness reporting.

### Scenario Design Rationale
- Scenario 1: Establish clean baseline without GAN synthesis.
- Scenario 2–3: Introduce controlled photometric & geometric variance to reduce overfitting to lighting and camera angle.
- Scenario 4: Increase structural variance (scaling/cropping) to improve localization robustness.
- Scenario 5: GAN synthetic images to mitigate residual minority class scarcity and improve small-pattern recognition.

Potential Extension: Evaluate domain adaptation (e.g., style transfer for different greenhouse lighting spectra) to further generalize beyond original facility conditions.

## 🧠 Proposed Architectures (YOLOV11 / YOLOV12)
Summarized architectural elements introduced in the paper:
- C2PSA: Cross-stage partial + self-attention fusion for richer multi-scale feature exchange with controlled compute.
- C3k2 / optimized C3 blocks: Reduced convolutional depth (two conv layers) to lower parameter count while retaining representational capacity.
- HFAM (Hierarchical Feature Attention Module): Channel + spatial reweighting across neck pyramid levels; boosts detection of small clustered deficiency regions.
- DCEL (Dynamic Context Enhancement Layer): Hybrid (SE + CBAM-inspired) attention improving robustness under heterogeneous lighting/background.
- Hybrid loss refinements: Emphasis on precise localization + class stability (still compatible with Ultralytics training interface).

Design intent: Preserve or improve mAP@50–95 while sustaining real-time FPS on mid-range GPU / potential edge accelerators.

### Attention & Efficiency Modules (Expanded)
- HFAM: Hierarchical aggregation of multi-scale neck outputs followed by channel & spatial attention; improves signal for subtle chlorosis edges.
- DCEL: Sequential squeeze-excitation + convolutional spatial attention blending, adaptively amplifying deficiency-specific discoloration under variable illumination.
- C2PSA vs classic CSP: Reduces redundant gradient paths while injecting self-attention for inter-region correlation (helpful when multiple deficiency symptoms co-occur).
- C3k2: Two-conv variant lowering FLOPs; empirical trade-off chosen to keep receptive field benefit while shrinking latency.

### Suggested Re-Implementation Path
If you wish to prototype YOLOV11/12 ideas inside this repo:
1. Fork Ultralytics repo (matching version used in paper timeframe).
2. Add custom module definitions (HFAM, DCEL) under `ultralytics/nn/modules/`.
3. Define new model YAML (extend from YOLOv8 base; insert attention blocks in neck).
4. Register losses or keep standard (paper indicates hybrid remains compatible).
5. Train with 224×224 first; profile FPS; then scale to 640 if accuracy gain justifies cost.

> Note: The actual canonical YOLOV11/12 code is not redistributed here—implementations must respect original licensing and paper authorship.

## 📈 Reported Paper Results (Excerpt)
Partial table excerpt (mAP values) for Scenarios 1–2 (two architectures). Higher numbers assigned per paper narrative to YOLOV12 (verify with full paper for complete table & statistical context):

| Scenario | Model | mAP@50 | mAP@50-95 |
|----------|-------|--------|-----------|
| 1 | YOLOV12 | 0.984 | 0.957 |
| 1 | YOLOV11 | 0.975 | 0.944 |
| 2 | YOLOV12 | 0.962 | 0.929 |
| 2 | YOLOV11 | 0.957 | 0.916 |

Disclaimer: Only a subset of results is transcribed from the provided text fragment; consult the paper PDF for full multi-scenario comparisons, speed (FPS), and resource usage analyses.

### Metric Definitions (Paper Terminology)
- train/box_loss: Bounding box regression error (localization quality).
- train/cls_loss: Classification term (class label correctness).
- train/dfl_loss: Distribution Focal Loss component for bounding box boundary distribution modeling.
- metrics/precision(B): Precision at chosen confidence / IoU.
- metrics/recall(B): Recall at chosen confidence / IoU.
- metrics/mAP50(B): Mean Average Precision at IoU 0.50.
- metrics/mAP50-95(B): Averaged mAP over IoU thresholds 0.50:0.05:0.95.
- F1_curve / P_curve / R_curve: Derived from varying confidence thresholds; useful to select deployment confidence.

Deployment Tip: Select confidence threshold maximizing F1 (balance) OR maximizing precision subject to minimum recall constraint relevant to agronomic intervention cost (e.g., <5% missed deficiencies).

## 📊 Final (Epoch 100) Metrics Snapshot
Taken directly from the last row of each `results.csv`.

| Model  | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|-----------|--------|-------|-----------|
| YOLOv5 | 0.94146   | 0.94558| 0.96065| 0.92039  |
| YOLOv6 | 0.94478   | 0.92206| 0.96676| 0.90114  |
| YOLOv8 | 0.93926   | 0.94819| 0.96166| 0.92194  |
| YOLOv9 | 0.93659   | 0.94514| 0.95694| 0.92125  |

Brief notes:
- YOLOv6 achieves the highest mAP@50 but has slightly lower mAP@50-95 than v8/v9.
- YOLOv8 & v9 are very close in overall mAP@50-95.
- YOLOv5 shows a balanced precision–recall profile.

The best epoch per model might not be the last; consult `results.png` or metric curves for early peaks.

## 🚀 Quick Start
1. Create environment (Python 3.10+ suggested):
	```bash
	python -m venv venv
	source venv/bin/activate  # Windows: venv\Scripts\activate
	pip install --upgrade pip
	```
2. Install core dependencies (example minimal):
	```bash
	pip install ultralytics==8.* torch torchvision torchaudio
	```
	For YOLOv5 / YOLOv6 official repos or forks also run: `pip install -r requirements.txt` where applicable.
3. Prepare dataset structure:
	```
	datasets/lettuce/
	  images/train  images/val
	  labels/train  labels/val
	```
	Example `data.yaml`:
	```yaml
	path: datasets/lettuce
	train: images/train
	val: images/val
	names: ["N_deficiency", "P_deficiency", "K_deficiency", "Healthy", "Other"]
	```
4. Open the desired notebook (`Trainyolov5.ipynb`, etc.) and execute sequentially.
5. Outputs appear under `runs/detect/train/`.

## 🛠 CLI Reproduction (YOLOv8 Example)
```bash
yolo detect train model=yolov8m.pt data=data.yaml epochs=100 imgsz=640 batch=40 project=runs name=train
```
Arguments may evolve across framework versions—prefer the notebooks for authoritative params.

## 📝 Notes & Assumptions
- Dataset is NOT bundled due to licensing/distribution constraints.
- Table values are from the final epoch, not necessarily the best checkpoint (e.g. `best.pt`).
- Folder name `yolovv9/` preserved as originally supplied.
- Paper PDF is included; detailed architectural diagrams, full scenario tables, FPS benchmarks, and ablations reside there.

## 📄 Citation
If you use this work, please cite the paper:
```
Bakir et al., 2025. Real‐Time Automatic Detection of Nutrient Deficiency in Lettuce Plants With New YOLOV12. Journal of Sensors, 2025.
```
And optionally the repository:
```
@software{lettuce_deficiency_yolo,
  author = {Cigdem Bakir,Ayberk Gezer},
  title  = {Real-Time Automatic Detection of Nutrient Deficiency in Lettuce Plants With New YOLOV12 model},
  year   = {2025},
  url    = {https://doi.org/10.1155/js/5592225}
}
```
Replace author list with the full, correct list from the paper.

## 📂 Artifacts
Per-model diagnostic assets in `runs/detect/train/`:
- `PR_curve.png`, `P_curve.png`, `R_curve.png`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `labels_correlogram.jpg`, `labels.jpg`
- `train_batch*.jpg` (augmentation samples)

## 🪪 License
Apache 2.0 (`LICENSE`). External pretrained weights (e.g. `yolov5mu.pt`, `yolov8m.pt`) are subject to their original repositories' licenses—verify before redistribution.
