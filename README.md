# NeuroChestYOLO: Dual-Stage Deep Learning Diagnostic Engine for Chest Radiographs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)](https://nextjs.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF1493)](https://github.com/ultralytics/ultralytics)

> **Live Web Application**: [https://neuro-chest-yolo.vercel.app](https://neuro-chest-yolo.vercel.app)  
> **Production API (Render)**: [https://neuro-cxr.onrender.com](https://neuro-cxr.onrender.com)  
> **GitHub Repository**: [https://github.com/debashis-sharingan-21/NeuroChestYOLO](https://github.com/debashis-sharingan-21/NeuroChestYOLO)  
> **Author**: [Debashis Mohapatra](https://www.linkedin.com/in/debashis-mohapatra-113523302/) ([debashismohapatra8260@gmail.com](mailto:debashismohapatra8260@gmail.com))

---

## 1. Executive Summary

**NeuroChestYOLO** is an end-to-end, production-grade clinical AI diagnostic platform designed to screen frontal chest radiographs (CXR) for 15 pulmonary, pleural, and cardiac pathologies in real time.

Most computer vision models in medical imaging suffer from either high false-positive spatial detections (e.g. standard object detectors hallucinating bounding boxes over ribs and hilar shadows) or lack of fine-grained spatial interpretability (e.g. black-box image classifiers). 

NeuroChestYOLO resolves this tension through a **Dual-Stage Gated Multimodal Architecture**:
1. **Stage 1 (Classification & Verification)**: A custom DenseNet-121 backbone equipped with a concatenated Dual-Pooling head (Global Average Pooling + Global Max Pooling) classifies 15 clinical findings simultaneously. Per-class decision thresholds calibrated via **Youden's J Index** determine statistical presence.
2. **Stage 2 (Spatial Grounding & Localization)**:
   - **Forward Linear Class Activation Mapping (CAM)** computes sub-3ms pixel-level saliency heatmaps directly from the DenseNet `norm5` feature maps and linear classifier weights, bypassing expensive backward autograd.
   - **Subordinated YOLOv8s Object Detection** operates strictly within conditions verified by the classifier gatekeeper. This prevents geometric hallucination over normal anatomical landmarks.
3. **Stage 3 (Clinical Risk Stratification)**: An automated risk-scoring engine compiles findings into a weighted clinical severity rating (CRITICAL, HIGH, MODERATE, LOW, NORMAL) to assist acute emergency triage.

---

## 2. System Architecture & Dual-Gating Pipeline

```
+--------------------------------------------------------------------------------------------------+
|                                     NEUROCHESTYOLO PIPELINE                                      |
+--------------------------------------------------------------------------------------------------+
                                          
                                  [ Frontal Chest Radiograph ]
                                               |
                                     (Preprocessing: 224px,
                                      ImageNet Mean/Std Norm)
                                               |
                                               v
                        +----------------------------------------------+
                        |           DenseNet-121 Backbone              |
                        |      (Backbone Feature Maps: norm5)          |
                        +----------------------------------------------+
                                          |          |
                   +----------------------+          +----------------------+
                   |                                                        |
                   v                                                        v
        +---------------------+                                  +---------------------+
        | AdaptiveAvgPool2d   |                                  |  AdaptiveMaxPool2d  |
        |     (1024-dim)      |                                  |     (1024-dim)      |
        +---------------------+                                  +---------------------+
                   |                                                        |
                   +----------------------+          +----------------------+
                                          |          |
                                          v          v
                        +----------------------------------------------+
                        |       Concatenated Feature Vector (2048-dim) |
                        +----------------------------------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        |        Linear Classification Head            |
                        |            (15 Output Logits)                |
                        +----------------------------------------------+
                                               |
                                        (Sigmoid Activation)
                                               |
                                               v
                        +----------------------------------------------+
                        |         Youden's J Optimal Thresholds        |
                        |         (Per-Class Decision Boundaries)      |
                        +----------------------------------------------+
                                               |
                          +--------------------+--------------------+
                          | (Probability >= Threshold)               | (Probability < Threshold)
                          v                                         v
               [ Confirmed Positive Findings ]              [ Ruled Out Conditions ]
                          |
         +----------------+----------------+
         |                                 |
         v                                 v
+-----------------------------+   +------------------------------------+
|  Forward Linear CAM Engine  |   |    Subordinated YOLOv8s Engine     |
|   (Features x Classifier W) |   |  (Conf >= 0.20, IoU NMS = 0.50)    |
|   - 0ms backward autograd   |   |  - Gated by positive findings      |
|   - Bilinear contour mask   |   |  - Spatial Bounding Boxes          |
+-----------------------------+   +------------------------------------+
         |                                 |
         +----------------+----------------+
                          |
                          v
         +------------------------------------------------+
         |          Clinical Risk Scoring Engine          |
         |  - Acuity Index: 0 to 100                     |
         |  - Triage Priority: CRITICAL / HIGH / MODERATE |
         |  - Critical Flags (e.g., Pneumothorax, Mass)   |
         +------------------------------------------------+
                          |
                          v
         +------------------------------------------------+
         |           Unified API Response Builder         |
         |  - Probabilities & Calibrated Thresholds       |
         |  - Base64 Overlays (Original, CAM, YOLO, Mix)  |
         |  - Sub-350ms End-to-End Latency                |
         +------------------------------------------------+
```

### Why Dual-Model Gating?
Standard object detectors (like YOLO) trained on annotated bounding boxes tend to over-detect patterns that resemble lesions, often identifying normal bronchial vascular markings or rib overlap as nodules or infiltrates. By forcing the YOLO detector to run under **DenseNet classification gating**, candidate bounding boxes are only accepted if the holistic DenseNet classifier independently validates the pathological presence above its calibrated threshold. This drastically reduces false-positive alarms while maintaining high localization sensitivity.

---

## 3. Real Empirical Benchmarks & Performance Metrics

All metrics reported below were computed directly against the trained model checkpoints (`models/nih-model.pth`, `models/best_chestxdet.pt`) and validated locally across test radiographs.

### A. Multi-Label Classification Performance (NIH ChestX-ray14 Validation Set)
- **Validation Mean ROC-AUC**: **0.7669 (76.69%)**
- **Validation Binary Cross-Entropy Loss**: **1.4446**
- **Optimal Decision Thresholds**: Calibrated via Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$).

| Pathology / Finding | Validation ROC-AUC | Optimal Decision Threshold | Priority Level |
| :------------------ | :----------------: | :------------------------: | :------------: |
| **Hernia** | **0.9009 (90.09%)** | 0.4578 | MODERATE |
| **Edema** | **0.8690 (86.90%)** | 0.3812 | HIGH |
| **Effusion** | **0.8588 (85.88%)** | 0.4120 | HIGH |
| **Cardiomegaly** | **0.8135 (81.35%)** | 0.4356 | HIGH |
| **Pneumothorax** | **0.7882 (78.82%)** | 0.3941 | CRITICAL |
| **Consolidation** | **0.7721 (77.21%)** | 0.3650 | HIGH |
| **Emphysema** | **0.7711 (77.11%)** | 0.4019 | MODERATE |
| **Atelectasis** | **0.7679 (76.79%)** | 0.3894 | MODERATE |
| **Fibrosis** | **0.7601 (76.01%)** | 0.3542 | LOW |
| **No Finding** | **0.7459 (74.59%)** | 0.5100 | NORMAL |
| **Mass** | **0.7131 (71.31%)** | 0.3420 | HIGH |
| **Pleural Thickening** | **0.7062 (70.62%)** | 0.3789 | LOW |
| **Pneumonia** | **0.6984 (69.84%)** | 0.3315 | HIGH |
| **Nodule** | **0.6844 (68.44%)** | 0.3204 | MODERATE |
| **Infiltration** | **0.6540 (65.40%)** | 0.3688 | MODERATE |
| **Macro Average (All 15 Classes)** | **0.7669 (76.69%)** | — | — |

---

### B. System Latency, Throughput & Memory Benchmark

Measured using [`scripts/benchmark.py`](scripts/benchmark.py) over 11 real clinical radiographs on standard single-thread CPU execution:

| Pipeline Stage | Mean Duration | Steady-State Latency | Description |
| :------------- | :-----------: | :------------------: | :---------- |
| **DenseNet-121 Forward Pass** | 29.50 ms | 26.10 ms | Multi-label 15-class inference under `torch.inference_mode()` |
| **Forward Linear CAM Generation** | 16.19 ms | 2.45 ms | Linear activation weighting without autograd backward passes |
| **YOLOv8s Pathology Detection** | 194.10 ms | 185.30 ms | Spatial localization, NMS filtering, bounding box rendering |
| **Response Assembly & Base64 Encode** | 89.40 ms | 82.10 ms | JSON building and 4x JPEG Base64 encoding |
| **End-to-End Pipeline Latency** | **348.01 ms** | **295.95 ms** | Complete pipeline from raw image bytes to full JSON/image response |
| **Inference Throughput** | **2.87 FPS** | **> 3.4 FPS** | Real-time interactive diagnostic performance |

#### Resident Memory (RAM RSS) Footprint:
- **Process Cold Start**: `443.46 MB`
- **Post-Model Registry Load**: `605.39 MB`
- **Peak Process RAM during Batch Inference**: `963.65 MB` (Zero memory leaks across consecutive inference cycles)

---

## 4. Repository Structure

```
NeuroChestYOLO/
├── backend/                          # FastAPI REST API & Deep Learning Pipeline
│   ├── main.py                       # Application lifecycle, CORS, /health and /predict endpoints
│   ├── model_loader.py               # Checkpoint loader, device abstraction, registry manager
│   ├── densenet_inference.py         # Forward pass & ultra-fast linear CAM generation
│   ├── yolo_inference.py             # Subordinated YOLOv8s detection & bounding box overlay
│   ├── risk_scorer.py                # Triage index calculation & acute clinical edge flags
│   ├── response_builder.py           # Unified JSON payload & base64 image compiler
│   └── utils.py                      # Image preprocessing (bilinear downsampling), Base64 encoders
│
├── frontend/                         # Next.js 16 (App Router + Turbopack) Clinical Terminal
│   ├── app/
│   │   ├── page.tsx                  # Interactive landing page & feature highlights
│   │   ├── dashboard/page.tsx        # Diagnostic terminal (Quad-view radiograph viewer, telemetry)
│   │   ├── performance/page.tsx      # Interactive ROC-AUC curves & benchmark breakdown
│   │   └── api/sample-images/        # Server route serving built-in clinical demo radiographs
│   ├── components/                   # Modular UI components (Navbar, Viewer, RiskBadge)
│   ├── lib/                          # State management & client utility functions
│   └── public/testing_images/        # Validated clinical sample radiographs
│
├── training/                         # Modular Re-Training & Evaluation Suite
│   ├── config.yaml                   # Hyperparameter configuration (batch size, lr, paths)
│   ├── dataset.py                    # Multi-label PyTorch Dataset with clinical augmentations
│   ├── model.py                      # Custom DenseNet-121 architecture with Dual Pooling Head
│   ├── train.py                      # Production training CLI with mixed precision & checkpointing
│   ├── evaluate.py                   # ROC-AUC evaluation & Youden's J threshold calibration CLI
│   └── test_api_endpoint.py          # Automated integration test suite for FastAPI endpoints
│
├── models/                           # Model weights & threshold calibration configs
│   ├── nih-model.pth                 # Trained DenseNet-121 PyTorch checkpoint
│   ├── best_chestxdet.pt             # Trained YOLOv8s pathology detection checkpoint
│   ├── thresholds_and_metrics.json   # Calibrated Youden J thresholds & validation metrics
│   └── yolo_pipeline_config.json     # Class overlap definitions & detection hyperparameters
│
├── scripts/
│   └── benchmark.py                  # Standalone local latency, throughput & memory profiler
│
├── Dockerfile                        # Multi-stage production container configuration
├── render.yaml                       # Cloud deployment configuration for Render
├── requirements.txt                  # Python dependencies pinned for reproducible runs
├── pyproject.toml                    # PEP 621 Python project metadata
└── LICENSE                           # Apache 2.0 Open-Source License
```

---

## 5. Quickstart & Local Setup

### Prerequisites
- **Python 3.10+ or 3.11** (Recommended: Miniconda / Anaconda)
- **Node.js 18+ or 20+** & `npm`
- **Git**

### Step 1: Clone the Repository
```bash
git clone https://github.com/debashis-sharingan-21/NeuroChestYOLO.git
cd NeuroChestYOLO
```

### Step 2: Backend Environment & Setup
```bash
# 1. Create and activate a dedicated Conda environment
conda create -n neuro python=3.11 -y
conda activate neuro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify backend integration tests
python training/test_api_endpoint.py
```

### Step 3: Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now accessible at `http://localhost:8000`. You can inspect interactive Swagger documentation at `http://localhost:8000/docs`.

### Step 4: Launch Next.js Diagnostic Terminal
Open a separate terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 6. Training, Evaluation & Profiling Workflows

### 1. Training from Scratch
To fine-tune or train the Dual-Pooling DenseNet-121 on your custom radiograph dataset:
```bash
# Using config.yaml:
python training/train.py --config training/config.yaml

# Or specify custom CLI arguments:
python training/train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --images_dir data/images \
  --batch_size 32 \
  --lr 0.0001 \
  --epochs 20 \
  --device cuda
```

### 2. Evaluating & Calibrating Youden's J Thresholds
To evaluate ROC-AUC performance and output calibrated decision boundaries:
```bash
python training/evaluate.py \
  --model models/nih-model.pth \
  --csv data/val.csv \
  --images_dir data/images \
  --output models/thresholds_and_metrics.json
```

### 3. Running the Local Performance Profiler
To measure component-level latency and peak RSS memory:
```bash
python scripts/benchmark.py
```

---

## 7. REST API Reference

### Health Check: `GET /health`
Returns the status of loaded models, active compute device, and available diagnostic classes.

**Sample Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Sample Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_classes": [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
  ],
  "device": "cpu"
}
```

---

### Radiograph Diagnostic Analysis: `POST /predict`
Performs end-to-end multi-label classification, forward CAM extraction, subordinated YOLO object detection, and risk triaging.

**Sample Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@frontend/public/testing_images/36204.png"
```

**Response Schema**:
```json
{
  "status": "success",
  "filename": "36204.png",
  "risk_score": 75.0,
  "risk_level": "HIGH",
  "case_flags": ["Mass", "Atelectasis"],
  "confirmed_findings": [
    {
      "condition": "Mass",
      "probability": 0.582,
      "threshold": 0.342,
      "confidence": "HIGH"
    },
    {
      "condition": "Atelectasis",
      "probability": 0.441,
      "threshold": 0.389,
      "confidence": "MODERATE"
    }
  ],
  "yolo_detections": [
    {
      "label": "Mass",
      "confidence": 0.612,
      "box_xyxy": [112.4, 74.8, 185.1, 148.3]
    }
  ],
  "images": {
    "original_b64": "data:image/jpeg;base64,...",
    "heatmap_b64": "data:image/jpeg;base64,...",
    "boxes_b64": "data:image/jpeg;base64,...",
    "combined_b64": "data:image/jpeg;base64,..."
  },
  "metrics": {
    "latency_seconds": 0.324,
    "top_condition": "Mass"
  }
}
```

---

## 8. Authorship & License

### Author & Engineering Lead
- **Debashis Mohapatra**
- LinkedIn: [https://www.linkedin.com/in/debashis-mohapatra-113523302/](https://www.linkedin.com/in/debashis-mohapatra-113523302/)
- GitHub: [@debashis-sharingan-21](https://github.com/debashis-sharingan-21)
- Email: [debashismohapatra8260@gmail.com](mailto:debashismohapatra8260@gmail.com)

### License
This project is open-sourced under the [Apache License 2.0](LICENSE).

---

> **Medical Disclaimer**: NeuroChestYOLO is an artificial intelligence research and educational software prototype. It is intended for investigational use and computer-aided triaging research only. It is not an FDA-cleared diagnostic medical device and should not replace clinical judgment or official radiological review by a licensed medical practitioner.
