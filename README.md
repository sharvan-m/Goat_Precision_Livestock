<div align="center">

# GoatVision: Precision Livestock Intelligence & Non-Invasive Biomarker Analytics

[![GitHub Repo](https://img.shields.io/badge/GitHub-sharvan--m%2FGoat__Precision__Livestock-blue?logo=github)](https://github.com/sharvan-m/Goat_Precision_Livestock)
[![Vision](https://img.shields.io/badge/Perception-YOLO11%20%7C%20SAM%20%7C%20ByteTrack-orange)](#-3-technology-stack)
[![Biometrics](https://img.shields.io/badge/Estimation-XGBoost%20%7C%20LightGBM%20%7C%20ICAR-blue)](#-3-technology-stack)
[![Health Diagnostics](https://img.shields.io/badge/Ocular%20Biomarker-FAMACHA%201--5%20Colorimetry-crimson)](#-3-technology-stack)
[![Platform](https://img.shields.io/badge/Full--Stack-Next.js%2016%20%7C%20FastAPI-green)](#-3-technology-stack)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#-license)

**An enterprise-grade, computer-vision and machine-learning framework for automated goat body segmentation, multi-animal ByteTrack tracking, intelligent side-view quality evaluation, allometric weight estimation, and non-invasive ocular FAMACHA anemia diagnostics.**

</div>

---

## 📖 1. Project Overview & Problem Statement

Small ruminant production (goats and sheep) is an essential economic cornerstone of global agriculture and pastoral food security. However, traditional livestock herd management relies extensively on manual, stressful, and physically demanding routines:

1. **Stressful Weight Scale Handling:** Forcing livestock across mechanical weigh scales induces acute handling stress, risks animal and operator injuries, alters metabolic rates, and proves impractical for high-frequency herd tracking.
2. **Fatal Parasitic Anemia Burden:** Gastrointestinal nematodes (predominantly *Haemonchus contortus*) cause acute anemia and high mortality in small ruminants. Manual **FAMACHA** eye scoring requires physical animal restraint and manual eyelid eversion by skilled technicians, making routine herd-wide screening logistically cost-prohibitive.
3. **Complex Unconstrained Farm Environments:** Dynamic animal crowding, self-occlusion, arbitrary camera viewpoints, and varying lighting conditions confound standard off-the-shelf object detectors.

```
                              ┌───────────────────────────────────┐
                              │ Input Video Stream / Sensor Image │
                              └─────────────────┬─────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │    YOLO11 / YOLO-World Detection  │
                              │    + ByteTrack Multi-Animal ID    │
                              └─────────────────┬─────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │  Morphology & Weight Path   │                   │  Ocular Anemia Health Path  │
        ├─────────────────────────────┤                   ├─────────────────────────────┤
        │ • Continuous View Quality   │                   │ • High-Precision Ocular     │
        │   Scoring (0 - 100)         │                   │   Conjunctiva Detection     │
        │ • Side-Profile Orientation  │                   │ • Mucosal Membrane ROI Mask │
        │ • Torso & Body SVD Axis     │                   │ • 5-Tier FAMACHA Classifier │
        │ • Metric Perspective Scale  │                   │ • CIELAB Colorimetric       │
        │ • Allometric ML Regression  │                   │   Spectrophotometry         │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │         GoatVision Suite          │
                              │  • FastAPI Asynchronous Engine    │
                              │  • Next.js 16 Analytic Dashboard  │
                              │  • Real-Time Video Telemetry      │
                              │  • Herd Database & Report Export  │
                              └───────────────────────────────────┘
```

**GoatVision** resolves these challenges by uniting cutting-edge computer vision (YOLO11, MobileSAM, Grounding DINO), multi-object tracking (ByteTrack), morphological biometrics (perspective-corrected body length, withers height, chest girth proxy, torso volume ratios), tree-based allometric regression (XGBoost, LightGBM, Random Forest), and deep ocular mucosal spectrophotometry into a unified full-stack system.

---

## ✨ 2. Key Features

- 🎯 **Multi-Stage Detection & Robust Segmentation:** Combines fine-tuned **YOLO11-seg**, **YOLOv8-seg**, and **MobileSAM** with zero-shot **YOLO-World** and **Grounding DINO** failover for reliable animal contour isolation in cluttered pens.
- 📐 **Intelligent 0–100 View-Quality Gate:** Automated view evaluation engine assesses body orientation angle, aspect ratio, contour solidity, border clipping, and occlusion overlap to ensure only true, un-occluded side profiles trigger weight estimation.
- ⚖️ **Non-Contact Allometric Weight Estimation:** Mathematical decomposition along the animal's principal body axis extracts physical proxies (body length, height, chest girth, torso area). Predictions are computed using veterinary allometric models (Schaeffer/ICAR standards) blended with trained XGBoost/LightGBM regressors.
- 🏷️ **Multi-Animal ByteTrack Tracking:** Real-time multi-target tracking assigns persistent IDs across video feeds, maintaining continuous temporal state, trajectory logging, and stable per-goat statistics.
- 🩸 **Non-Invasive Ocular FAMACHA Anemia Screening:** Automated ocular conjunctiva localization coupled with a 5-class mucosal color classifier (`Nivel 1`–`Nivel 5`) and CIELAB colorimetry to quantify anemia risk without drawing blood.
- 🖥️ **Full-Stack Enterprise Web Application:** Production-grade Next.js 16 (React 19, TypeScript, Tailwind CSS) user interface integrated with a high-performance FastAPI asynchronous backend supporting live webcam feeds, batch video uploads, and RESTful telemetry.
- 📊 **Herd Analytics & Audit Exports:** Automated record ingestion with built-in export capabilities to CSV and Excel (`.xlsx`) formats for livestock management records.

---

## 🛠️ 3. Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend Framework** | Next.js 16 (App Router) / React 19 / TypeScript | Reactive, dark-themed dashboard, real-time analytics & media rendering |
| **UI & Styling** | Tailwind CSS / Lucide React | Modern responsive design, SVG telemetry gauges, and clean status indicators |
| **Backend Framework** | FastAPI / Starlette / Uvicorn | High-throughput asynchronous REST API, background job queues, media static serving |
| **Core Deep Learning** | PyTorch / Torchvision / CUDA | GPU-accelerated tensor computation and deep neural network inference |
| **Object Detection & Seg** | Ultralytics YOLO11 / YOLOv8 / MobileSAM | Instance segmentation of goat bodies, torso regions, and ocular conjunctiva |
| **Zero-Shot Perception** | YOLO-World / Grounding DINO | Robust fallback detection for edge cases and unconstrained farm lighting |
| **Multi-Object Tracking** | ByteTrack (`supervision 0.27.0`) | Associative bounding-box tracking and identity persistence across video frames |
| **Machine Learning Suite** | XGBoost / LightGBM / Scikit-Learn | Morphological feature regression, multi-model ensemble weight estimation |
| **Image Processing** | OpenCV / Pillow / SciPy / NumPy | SVD principal axis extraction, contour moments, CIELAB colorimetric analysis |
| **Database & Export** | Pandas / OpenPyXL / Joblib | Herd registry serialization, pipeline artifact loading, CSV/Excel reporting |

---

## 📂 4. Repository Structure

```text
Goat_Precision_Livestock/
├── backend/                             # Core FastAPI application & AI service layer
│   ├── routes/                          # Modular API route controllers
│   │   └── video_analysis.py            # Long-form video ingestion & batch inference endpoints
│   ├── services/                        # Specialized business & model execution logic
│   │   ├── anemia_famacha.py            # FAMACHA 1-5 ocular classification & colorimeter
│   │   ├── anemia_pipeline.py           # Two-stage conjunctiva detection + scoring orchestrator
│   │   ├── conjunctiva_local.py         # Local YOLO conjunctiva detector
│   │   ├── conjunctiva_roboflow.py      # Cloud Roboflow conjunctiva workflow client
│   │   ├── famacha_colorimeter.py       # CIELAB & RGB mucosal spectrophotometric analysis
│   │   ├── famacha_local.py             # Local FAMACHA classifier execution
│   │   ├── side_classifier.py           # Dedicated side-profile orientation classifier
│   │   ├── torso_segmenter.py           # Torso-specific mask decomposition logic
│   │   └── roboflow_workflow.py         # Remote Roboflow API integration layer
│   ├── main.py                          # Primary FastAPI server entry point & route definitions
│   ├── requirements.txt                 # Backend Python package dependencies
│   └── .env.example                     # Environment template (API keys, ports, model paths)
│
├── frontend/                            # Modern Next.js 16 Web Dashboard
│   ├── app/                             # Next.js App Router pages
│   │   ├── detect/page.tsx              # Single-image detection, side-view & weight analysis
│   │   ├── analyze/page.tsx             # Video analysis pipeline with frame scrubbing
│   │   ├── tracking/page.tsx            # Real-time multi-goat ByteTrack telemetry
│   │   ├── anemia/page.tsx              # Ocular conjunctiva FAMACHA anemia diagnostic suite
│   │   ├── records/page.tsx             # Herd management database, table search & exports
│   │   ├── system/page.tsx              # System diagnostics, model status & calibration tools
│   │   ├── layout.tsx                   # Global application layout shell
│   │   └── globals.css                  # Custom styling & design tokens
│   ├── components/                      # Reusable UI components (Sidebar, UploadDropzone, etc.)
│   ├── lib/                             # Frontend API clients, state context & TypeScript definitions
│   └── package.json                     # Node.js dependencies & scripts
│
├── goat_weight_system/                  # Autonomous morphological weight estimation package
│   ├── config.py                        # System constants, view score thresholds, class mappings
│   ├── detector.py                      # Unified GoatDetector interface (YOLO / SAM)
│   ├── tracker.py                       # Individual bounding-box tracking module
│   ├── bytetrack_system.py              # Full ByteTrack multi-target pipeline integration
│   ├── view_evaluator.py                # 0-100 View Quality Scoring algorithm
│   ├── measurement_extractor.py         # Morphological feature extractor (BL, WH, CG, Torso)
│   ├── regression_models.py             # XGBoost/LightGBM/RF weight regressor suite & ICAR formulas
│   ├── evaluation.py                    # Statistical confidence metrics & error bounds
│   ├── visualizer.py                    # Bounding box, contour mask & HUD telemetry overlay
│   ├── data_logger.py                   # Automated CSV / XLSX herd registry persistence
│   └── video_processor.py               # Batch video processing pipeline
│
├── Anemia/                              # Specialized ocular anemia research assets
│   └── goat_eye_detection/              # Trained YOLO weights for eye & conjunctiva extraction
│       ├── yolo11n-cls.pt               # Ocular classification checkpoint
│       └── model/
│           ├── conjunctiva_best.pt      # Fine-tuned conjunctiva segmentation model
│           └── famacha_best.pt          # Fine-tuned 5-class FAMACHA classification model
│
├── models/                              # Serialized model weights & regression artifacts
│   ├── best_weight_model.pkl            # Production trained tree-based weight regressor
│   └── goat_yolo_seg.pt                 # Optimized goat instance segmentation weights
│
├── goat_metric_yw/                      # Metric regression pipelines & assist models (.joblib)
├── goat_rgb_full_pipeline/              # End-to-end RGB pipeline reports & cross-validation runs
├── tests/                               # Comprehensive automated test suite
│   ├── test_anemia_pipeline.py          # Pytest suite for two-stage anemia gating & scoring
│   └── test_detector_selection.py      # Regression tests for detection fallback logic
│
├── goat_detect.py                       # Standalone zero-shot & SAM segmentation utility
├── goat_video_analyzer.py               # Standalone multi-feature video analyzer CLI
├── infer_goat_weight.py                 # Standalone single-frame weight inference CLI
└── README.md                            # Comprehensive project documentation
```

---

## 🔬 5. Architectural Deep Dive & Methodology

### 5.1 Morphological Weight Estimation Pipeline

The weight of an animal is physically correlated with its skeletal and volumetric dimensions according to standard zootechnical allometry:

$$\text{Weight} \approx k \cdot (\text{Body Length})^{\alpha} \cdot (\text{Chest Girth})^{\beta} \cdot (\text{Withers Height})^{\gamma}$$

GoatVision measures these parameters from monocular or calibrated RGB/RGB-D camera feeds:

1. **Contour Analysis & Principal Orientation:** The binary segmentation mask $M$ is converted to coordinate point sets $(x_i, y_i)$. Singular Value Decomposition (SVD) or Principal Component Analysis (PCA) identifies the primary elongation axis, isolating head-to-tail orientation from limb excursions.
2. **Biometric Extraction:**
   - **Body Length ($BL$):** Distance along the principal axis between the cranial point (shoulder/wither) and the caudal point (pin bone).
   - **Withers Height ($WH$):** Vertical projection perpendicular to the ground plane from hoof baseline to the dorsal spinal ridge.
   - **Chest Girth Proxy ($CG$):** Torso cross-sectional diameter estimated via torso segmenter mask width.
   - **Torso Area Ratio:** The proportion of total silhouette occupied by the thoracic-abdominal cavity.
3. **Multi-Model Regression Ensemble:** Extracted dimensions feed into trained regressors (XGBoost, LightGBM, Gradient Boosting, and Log-Linear Models) trained against ground-truth livestock scale measurements.

### 5.2 Flexible 0–100 View-Quality Scoring Engine

Single-camera weight estimation requires the animal to present a perpendicular side profile. GoatVision computes a continuous **View Quality Score (0–100)**:

$$\text{Score} = w_1 S_{\text{vis}} + w_2 S_{\text{orient}} + w_3 S_{\text{seg}} + w_4 S_{\text{size}} + w_5 S_{\text{clarity}} - P_{\text{border}} - P_{\text{occlusion}}$$

| Quality Tier | Score Range | Annotation Color | Weight Estimation Gate | Telemetry Output |
| :---: | :---: | :---: | :---: | :---: |
| **GOOD** | $\ge 75.0$ | **Green** | **Enabled** | High confidence reading; record committed to database. |
| **ACCEPTABLE** | $55.0 - 74.9$ | **Yellow** | **Enabled** | Medium confidence; warning issued on extreme aspect deviations. |
| **POOR** | $< 55.0$ | **Red** | **Disabled** | Rejection logged (e.g. frontal angle, grazing pose, border cut). |

### 5.3 Non-Invasive Ocular FAMACHA Health Pipeline

The **FAMACHA** system evaluates the mucous membrane color of the ocular conjunctiva to grade anemia severity on a 1 to 5 scale:

```
[Raw Head/Eye Crop] ──► [Conjunctiva Detection] ──► [Mucosal Mask Extraction] ──► [FAMACHA Classifier / CIELAB Colorimetry]
```

- **Score 1 (Optimal):** Cherry red mucous membrane ($\text{Hematocrit} \ge 28\%$). No anemia; treatment not required.
- **Score 2 (Acceptable):** Red-pink mucous membrane ($\text{Hematocrit} \approx 23 - 27\%$). Healthy animal.
- **Score 3 (Borderline):** Pink mucous membrane ($\text{Hematocrit} \approx 18 - 22\%$). Monitor closely.
- **Score 4 (Dangerous):** Pink-white mucous membrane ($\text{Hematocrit} \approx 13 - 17\%$). Anemic; immediate anthelmintic dosing recommended.
- **Score 5 (Severe / Fatal):** Pure white/blanched mucous membrane ($\text{Hematocrit} < 13\%$). Critical medical emergency.

The pipeline integrates automated confidence gating: if the conjunctiva cannot be localized with high confidence ($\ge 0.50$), the system rejects the image rather than issuing an unreliable diagnostic score.

---

## ⚡ 6. Installation & Setup

### Environment Requirements
- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS
- **Python:** 3.10 or 3.11
- **Node.js:** 18.17+ or 20.x
- **CUDA (Optional but Recommended):** NVIDIA GPU with CUDA Compute $\ge 7.5$ (CUDA 11.8 or 12.x)

### 1. Clone the Repository
```bash
git clone https://github.com/sharvan-m/Goat_Precision_Livestock.git
cd Goat_Precision_Livestock
```

### 2. Backend Environment Setup
Create and activate an isolated Python virtual environment:
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Install backend dependencies:
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

*(Optional)* Configure local environment variables:
```bash
cp backend/.env.example backend/.env
```

### 3. Frontend Environment Setup
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
cd ..
```

---

## 🚀 7. Running the System

### 1. Launch the Backend API
Start the FastAPI server from the project root:
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Swagger Interactive Documentation: `http://localhost:8000/docs`
- ReDoc Documentation: `http://localhost:8000/redoc`

### 2. Launch the Next.js Frontend
In a separate terminal, start the Next.js development server:
```bash
cd frontend
npm run dev
```
Open your browser at `http://localhost:3000`.

### 3. Standalone CLI & Diagnostic Tools

**Run Single-Image Morphological Weight Inference:**
```bash
python infer_goat_weight.py path/to/goat_rgb.jpg path/to/depth_map.png main
```

**Run Full Video Analysis with Automated View Selection:**
```bash
python goat_video_analyzer.py
```

**Execute Automated Test Suite:**
```bash
pytest tests/ -v
```

---

## 📡 8. REST API Reference

The FastAPI backend exposes modular REST endpoints for seamless integration with IoT cameras and edge gateways:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and hardware acceleration check |
| `POST` | `/api/detect/goats` | Single-image detection, instance mask segmentation, view quality check & weight estimation |
| `POST` | `/api/detect/recalibrate` | Dynamic recalibration of the pixel-to-metric scaling factor |
| `POST` | `/api/analyze/image` | Morphological feature extraction and multi-model weight prediction |
| `POST` | `/api/analyze/video` | Asynchronous video processing job creation with automated best-frame selection |
| `GET` | `/api/jobs/{job_id}` | Query the progress and results of an ongoing video analysis job |
| `POST` | `/api/tracking/video` | Multi-animal ByteTrack video processor returning annotated frames and trajectory logs |
| `GET` | `/api/anemia/status` | Readiness and version telemetry of conjunctiva and FAMACHA models |
| `POST` | `/api/anemia/analyze` | End-to-end ocular conjunctiva segmentation and FAMACHA 1–5 mucosal classification |
| `GET` | `/api/records` | Retrieve historical livestock weight logs and clinical records (CSV / XLSX format) |
| `GET` | `/api/analytics` | Statistical summaries, herd weight distributions, and health risk indicators |
| `GET` | `/api/media` | Secure static media file retrieval for processed video clips and visual overlays |

---

## 📋 9. Project Implementation Status

- [x] Multi-model goat detection and segmentation pipeline (YOLO11, YOLOv8-seg, MobileSAM).
- [x] Continuous 0–100 view quality scoring algorithm with orientation and occlusion rejection.
- [x] Multi-target ByteTrack integration with ID persistence across video streams.
- [x] Biometric feature extraction along principal body axes (BL, WH, CG, Torso Volume).
- [x] Allometric weight estimation engine (XGBoost, LightGBM, Random Forest, ICAR equations).
- [x] Two-stage ocular conjunctiva localization and 5-class FAMACHA anemia classification.
- [x] Full-stack enterprise web dashboard (Next.js 16, React 19, TypeScript, Tailwind CSS).
- [x] Automated test suite covering anemia gating logic and fallback detectors.
- [ ] Edge deployment support for NVIDIA Jetson (TensorRT engine optimization).
- [ ] Integration with RFID ear-tag readers for automatic digital twin synchronization.
- [ ] Multi-camera stereoscopic 3D point-cloud surface mesh reconstruction.

---

## 📚 10. References & Scientific Literature

1. **FAMACHA Anemia Diagnostic System:**  
   van Wyk, J. A., & Bath, G. F. (2002). *The FAMACHA system for managing haemonchosis in sheep and goats by clinically evaluating anaemia.* Veterinary Parasitology, 108(4), 273–276.
2. **ByteTrack Multi-Object Tracking:**  
   Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* European Conference on Computer Vision (ECCV).
3. **Segment Anything (SAM):**  
   Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023). *Segment Anything.* IEEE/CVF International Conference on Computer Vision (ICCV).
4. **Livestock Morphometrics & Allometry:**  
   International Committee for Animal Recording (ICAR). *Guidelines for Conformation Recording and Linear Scoring in Small Ruminants.*
5. **Ultralytics YOLO Architecture:**  
   Jocher, G., Qiu, J., et al. (2024). *Ultralytics YOLO11: State-of-the-art Real-time Computer Vision.* [Ultralytics Hub](https://github.com/ultralytics/ultralytics).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
