<div align="center">

# GoatVision: Precision Livestock Intelligence & Non-Invasive Biomarker Analytics

[![GitHub Repo](https://img.shields.io/badge/GitHub-sharvan--m%2FGoat__Precision__Livestock-blue?logo=github)](https://github.com/sharvan-m/Goat_Precision_Livestock)
[![Vision](https://img.shields.io/badge/Perception-YOLO11%20%7C%20SAM%20%7C%20ByteTrack-orange)](#-3-technology-stack)
[![Biometrics](https://img.shields.io/badge/Estimation-XGBoost%20%7C%20LightGBM%20%7C%20ICAR-blue)](#-3-technology-stack)
[![Health Diagnostics](https://img.shields.io/badge/Diagnostics-FAMACHA%201--5%20Colorimetry-crimson)](#52-non-invasive-ocular-famacha-pipeline)
[![Platform](https://img.shields.io/badge/Full--Stack-Next.js%2016%20%7C%20FastAPI-green)](#-3-technology-stack)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#-license)

**An enterprise-grade precision livestock framework combining real-time computer vision, multi-animal ByteTrack tracking, automated side-profile view evaluation, biometric morphological weight estimation, and non-invasive ocular FAMACHA anemia diagnostics.**

[Overview](#-1-project-overview) • [Key Features](#-2-key-features) • [Tech Stack](#-3-technology-stack) • [Repository Structure](#-4-repository-structure) • [Architecture](#-5-architecture--methodology) • [Quick Start](#-6-quick-start) • [API Reference](#-7-rest-api-reference) • [References](#-8-references--literature)

</div>

---

## 📖 1. Project Overview

Manual livestock management creates major bottlenecks in modern caprine and ovine farming:
- **Stress & Injury:** Mechanical scales induce acute animal stress and injury risks for handlers.
- **Parasite Burden:** *Haemonchus contortus* (barber's pole worm) causes lethal **anemia**. Traditional **FAMACHA** eyelid inspections require manual physical restraint, making routine herd coverage labor-intensive.
- **Farm Occlusions:** Crowded pens, overlapping animals, and non-ideal angles confound off-the-shelf detectors.

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
        │ • View Quality Gate (0-100) │                   │ • Ocular Conjunctiva Mask   │
        │ • SVD Principal Body Axis   │                   │ • Mucosal Membrane ROI Crop │
        │ • Real-Scale Perspective    │                   │ • 5-Tier FAMACHA Classifier │
        │ • XGBoost/LightGBM Weights  │                   │ • CIELAB Colorimetry        │
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │     FastAPI + Next.js 16 Suite    │
                              │  • Real-Time Video Telemetry      │
                              │  • Herd Database & CSV/XLSX Logs  │
                              └───────────────────────────────────┘
```

**GoatVision** automates this entirely: zero-contact **body weight estimation** from side-profile silhouettes and non-invasive **ocular conjunctiva health diagnostics** through an end-to-end full-stack platform.

---

## ✨ 2. Key Features

- **🎯 Dual-Stage Instance Segmentation:** Fine-tuned **[YOLO11-seg](https://github.com/ultralytics/ultralytics)** and **[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)** with **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)** zero-shot fallback for robust animal isolation in unconstrained pens.
- **📐 0–100 View-Quality Gate:** Continuously grades animal orientation, solidity, border clipping, and occlusion to ensure **only true side profiles** trigger weight calculations.
- **⚖️ Non-Contact Weight Regression:** Computes perspective-corrected **Body Length**, **Withers Height**, **Chest Girth proxy**, and **Torso Volume**, feeding **XGBoost** and **LightGBM** regressors mapped to **ICAR** allometrics.
- **🏷️ Multi-Animal ByteTrack Tracking:** Real-time multi-target tracking via **[ByteTrack](https://github.com/ifzhang/ByteTrack)** assigns persistent animal IDs with continuous trajectory logging across camera feeds.
- **🩸 Ocular FAMACHA Anemia Screening:** Detects the eye conjunctiva and scores mucosal pallor across **5 clinical tiers** (`Nivel 1`–`Nivel 5`) using deep classification and **CIELAB colorimetry**.
- **🖥️ Enterprise Web Dashboard:** Modern, reactive **[Next.js 16](https://nextjs.org)** interface (React 19, TypeScript, Tailwind CSS) paired with an asynchronous **[FastAPI](https://fastapi.tiangolo.com)** backend.
- **📊 Herd Analytics & Auditing:** Instant export of herd health, weight distribution, and historical logs to **CSV** and **Excel (`.xlsx`)**.

---

## 🛠️ 3. Technology Stack

| Component | Framework / Tool | Purpose |
| :--- | :--- | :--- |
| **Frontend** | [Next.js 16](https://nextjs.org) / [React 19](https://react.dev) / [TypeScript](https://www.typescriptlang.org) | Reactive dashboard, live telemetry, and media viewer |
| **Styling** | [Tailwind CSS](https://tailwindcss.com) / [Lucide Icons](https://lucide.dev) | Responsive UI, dark theme, and visual health status badges |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com) / [Uvicorn](https://www.uvicorn.org) / [Pydantic](https://docs.pydantic.dev) | High-throughput asynchronous REST engine and background workers |
| **Vision & Seg** | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) / [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Body, torso, and ocular conjunctiva instance segmentation |
| **Zero-Shot Detect** | [YOLO-World](https://github.com/AILab-CVC/YOLO-World) / [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Robust open-vocabulary fallback for unconstrained pen conditions |
| **Target Tracking** | [ByteTrack](https://github.com/ifzhang/ByteTrack) via [Supervision](https://supervision.roboflow.com) | Multi-animal tracking and ID re-identification |
| **ML Regression** | [XGBoost](https://xgboost.readthedocs.io) / [LightGBM](https://lightgbm.readthedocs.io) / [Scikit-Learn](https://scikit-learn.org) | Morphological biometrics to body weight (kg) prediction |
| **Image Analysis** | [OpenCV](https://opencv.org) / [SciPy](https://scipy.org) / [NumPy](https://numpy.org) | SVD principal axis calculation, contour moments, and CIELAB color |

---

## 📂 4. Repository Structure

```text
Goat_Precision_Livestock/
├── backend/                             # FastAPI application & AI service layer
│   ├── routes/video_analysis.py         # Video upload & batch processing endpoints
│   ├── services/
│   │   ├── anemia_famacha.py            # FAMACHA 1-5 mucosal classifier & colorimeter
│   │   ├── anemia_pipeline.py           # Two-stage conjunctiva detection + scoring
│   │   ├── conjunctiva_local.py         # Local YOLO conjunctiva detector
│   │   ├── famacha_colorimeter.py       # CIELAB & RGB mucosal spectrophotometry
│   │   ├── side_classifier.py           # Side-profile view orientation classifier
│   │   └── torso_segmenter.py           # Torso-specific mask isolation
│   ├── main.py                          # Primary API gateway & REST endpoints
│   └── requirements.txt                 # Backend Python package dependencies
│
├── frontend/                            # Next.js 16 App Router dashboard
│   ├── app/
│   │   ├── detect/page.tsx              # Single-image detection & weight analysis
│   │   ├── analyze/page.tsx             # Video analysis pipeline with frame scrubber
│   │   ├── tracking/page.tsx            # Live multi-goat ByteTrack telemetry
│   │   ├── anemia/page.tsx              # FAMACHA ocular anemia diagnostic suite
│   │   ├── records/page.tsx             # Herd management database & CSV/XLSX exports
│   │   └── system/page.tsx              # Model status & metric calibration settings
│   ├── components/                      # UI modules (Sidebar, UploadDropzone, etc.)
│   └── package.json                     # Frontend dependencies & npm scripts
│
├── goat_weight_system/                  # Morphological weight estimation package
│   ├── detector.py                      # Unified YOLO / SAM detection interface
│   ├── bytetrack_system.py              # Multi-target ByteTrack integration
│   ├── view_evaluator.py                # 0-100 View Quality Scoring algorithm
│   ├── measurement_extractor.py         # Morphological feature extractor (BL, WH, CG, Torso)
│   ├── regression_models.py             # XGBoost/LightGBM regressors & ICAR allometrics
│   └── data_logger.py                   # Automated CSV / Excel herd records writer
│
├── Anemia/goat_eye_detection/model/     # Checkpoints: conjunctiva_best.pt, famacha_best.pt
├── models/                              # Serialized models: best_weight_model.pkl, goat_yolo_seg.pt
├── tests/                               # Automated test suite (pytest)
│   ├── test_anemia_pipeline.py          # Two-stage anemia pipeline validation
│   └── test_detector_selection.py      # Fallback detector unit tests
│
├── goat_detect.py                       # Standalone zero-shot & SAM segmentation utility
├── goat_video_analyzer.py               # Standalone multi-feature video analyzer CLI
├── infer_goat_weight.py                 # Standalone single-frame weight inference CLI
└── README.md                            # Project documentation
```

---

## 🔬 5. Architecture & Methodology

### 5.1 Weight Estimation & 0–100 View Quality Gate

Weight is modeled using zootechnical allometrics: $\text{Weight} \approx k \cdot (\text{Body Length})^{\alpha} \cdot (\text{Chest Girth})^{\beta} \cdot (\text{Withers Height})^{\gamma}$.

To ensure measurement validity, incoming frames pass through an automated **0–100 View Quality Gate**:

$$\text{Score} = w_1 S_{\text{vis}} + w_2 S_{\text{orient}} + w_3 S_{\text{seg}} + w_4 S_{\text{size}} + w_5 S_{\text{clarity}} - P_{\text{border}} - P_{\text{occlusion}}$$

| Quality Tier | Score Range | Status | Action |
| :---: | :---: | :---: | :--- |
| **GOOD** | $\ge \mathbf{75.0}$ | **Green** | **Full estimation enabled**; record committed to database. |
| **ACCEPTABLE** | $\mathbf{55.0 - 74.9}$ | **Yellow** | **Estimated with medium confidence**; logs view warnings. |
| **POOR** | $< \mathbf{55.0}$ | **Red** | **Rejected** (e.g. non-side angle, grazing pose, border clip). |

### 5.2 Non-Invasive Ocular FAMACHA Pipeline

Screening operates in a two-stage cascaded pipeline:

```
[Eye / Head Image] ──► [Conjunctiva Detection (YOLO)] ──► [Mucosal Mask Crop] ──► [FAMACHA 1-5 Classifier]
```

- **Score 1 (Optimal):** Cherry red membrane ($\text{Hematocrit} \ge 28\%$). **Healthy; no dosing required.**
- **Score 2 (Acceptable):** Red-pink membrane ($\text{Hematocrit} \approx 23 - 27\%$). **Healthy animal.**
- **Score 3 (Borderline):** Pink membrane ($\text{Hematocrit} \approx 18 - 22\%$). **Monitor closely.**
- **Score 4 (Dangerous):** Pink-white membrane ($\text{Hematocrit} \approx 13 - 17\%$). **Higher risk; anthelmintic dose recommended.**
- **Score 5 (Severe):** Pure white membrane ($\text{Hematocrit} < 13\%$). **Critical emergency treatment required.**

*Gate Rule: If conjunctiva localization confidence is below **0.50**, the diagnostic is rejected to prevent false scores.*

---

## ⚡ 6. Quick Start

### 1. Clone & Set Up Backend
```bash
git clone https://github.com/sharvan-m/Goat_Precision_Livestock.git
cd Goat_Precision_Livestock

# Python virtual environment (Python 3.10+)
python -m venv venv
.\venv\Scripts\Activate.ps1       # On Linux/macOS: source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Set Up Frontend
```bash
cd frontend
npm install
cd ..
```

### 3. Launch Services
**Terminal 1 — Backend API (FastAPI):**
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
*API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)*

**Terminal 2 — Web Dashboard (Next.js):**
```bash
cd frontend
npm run dev
```
*Web App: [http://localhost:3000](http://localhost:3000)*

### 4. Standalone CLI & Tests
```bash
# Single image weight estimation
python infer_goat_weight.py path/to/goat.jpg path/to/depth.png main

# Video analysis
python goat_video_analyzer.py

# Run test suite
pytest tests/ -v
```

---

## 📡 7. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and CUDA acceleration telemetry |
| `POST` | `/api/detect/goats` | Single-image detection, mask segmentation, view scoring & weight |
| `POST` | `/api/detect/recalibrate` | Dynamic recalibration of the pixel-to-metric scaling factor |
| `POST` | `/api/analyze/image` | Morphological feature extraction and allometric prediction |
| `POST` | `/api/analyze/video` | Asynchronous video processing with automated best-frame selection |
| `GET` | `/api/jobs/{job_id}` | Poll asynchronous video job status and results |
| `POST` | `/api/tracking/video` | Multi-animal ByteTrack processing with persistent IDs and trajectory logs |
| `GET` | `/api/anemia/status` | Model status for conjunctiva detector and FAMACHA classifier |
| `POST` | `/api/anemia/analyze` | Two-stage ocular conjunctiva segmentation and FAMACHA 1–5 scoring |
| `GET` | `/api/records` | Export historical logs in **CSV** or **Excel (`.xlsx`)** format |
| `GET` | `/api/analytics` | Herd weight distributions, averages, and clinical risk breakdown |

---

## 📋 8. Project Implementation Status

- [x] Multi-model goat detection and segmentation (**YOLO11**, **YOLOv8-seg**, **MobileSAM**).
- [x] Continuous **0–100 view quality scoring gate** with angle and occlusion rejection.
- [x] Multi-target **ByteTrack tracking** with ID persistence across video streams.
- [x] Biometric feature extraction along principal body axes (**BL**, **WH**, **CG**, **Torso Volume**).
- [x] Allometric weight estimation engine (**XGBoost**, **LightGBM**, **ICAR** equations).
- [x] Two-stage ocular conjunctiva localization and **5-class FAMACHA** anemia grading.
- [x] Full-stack enterprise web dashboard (**Next.js 16**, **React 19**, **TypeScript**, **Tailwind CSS**).
- [x] Automated test suite covering anemia gating logic and fallback detectors.
- [ ] Edge optimization for NVIDIA Jetson deployment (TensorRT engines).
- [ ] RFID ear-tag sensor pairing for digital herd record synchronization.

---

## 📚 9. References & Literature

1. **FAMACHA Anemia Diagnostic System:**  
   van Wyk, J. A., & Bath, G. F. (2002). *The FAMACHA system for managing haemonchosis in sheep and goats by clinically evaluating anaemia.* Veterinary Parasitology, 108(4), 273–276.  
   📄 [DOI: 10.1016/S0304-4017(02)00267-7](https://doi.org/10.1016/S0304-4017(02)00267-7) • 🌐 [WormsX FAMACHA Guide](https://www.wormx.info/famacha)

2. **ByteTrack Multi-Object Tracking:**  
   Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.  
   📄 [arXiv:2110.06864](https://arxiv.org/abs/2110.06864) • 💻 [GitHub: ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)

3. **Segment Anything (SAM) & MobileSAM:**  
   Kirillov, A., et al. (2023). *Segment Anything.* ICCV 2023.  
   Zhang, C., et al. (2023). *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications.*  
   📄 [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) • 📄 [arXiv:2306.14289](https://arxiv.org/abs/2306.14289) • 💻 [GitHub: MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

4. **Zero-Shot Object Detection (Grounding DINO & YOLO-World):**  
   Liu, S., et al. (2023). *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.*  
   📄 [arXiv:2303.05499](https://arxiv.org/abs/2303.05499) • 💻 [GitHub: GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) • 💻 [GitHub: YOLO-World](https://github.com/AILab-CVC/YOLO-World)

5. **Livestock Morphometrics & Allometry Standards:**  
   International Committee for Animal Recording (ICAR). *Guidelines for Conformation Recording and Linear Scoring in Small Ruminants.*  
   🌐 [ICAR Official Guidelines](https://www.icar.org/)

6. **Ultralytics YOLO Architecture:**  
   Jocher, G., Qiu, J., et al. (2024). *Ultralytics YOLO11: Real-time Computer Vision.*  
   💻 [GitHub: Ultralytics](https://github.com/ultralytics/ultralytics) • 🌐 [Documentation](https://docs.ultralytics.com/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
