# FAST — National University of Computer and Emerging Sciences

**FAST School of Computing · Karachi Campus · Department of Software Engineering**
**AL2002 Artificial Intelligence (Laboratory) · Fall 2026**

# PROJECT PROPOSAL

*Semester Lab Project · Weightage 1 mark*

---

## Group Information

| Name | Roll Number | Section | Role in the project |
|---|---|---|---|
| Muhammad Ali (Leader) | 24K-3103 | BSE-5B | Architecture, AI Worker Pipeline, Tracking & Concurrency |
| Mohammad Hassan | 24K-3051 | BSE-5B | Model training, model integration, WebSocket integration, deployment |
| Asad Chotani | 24K-3072 | BSE-5B | Frontend Dashboard, WebSocket Streaming & Evaluation Benchmarking |

| Field | Detail |
|---|---|
| **Project Title** | Automated Multi-Camera Object Detection, Tracking, and Inventory Counting System for Industrial RTSP Streams |
| **Project Idea** | Idea No. — from the Project Ideas list ☐ Own idea (needs approval) ☑ |
| **Category** | Part A — Game ☐ Part B — Applied / Data ☑ |
| **GitHub Repository URL** | https://github.com/mhassan25-ops/Object-Detection-AI.git |
| **Date of Submission** | October 20, 2026 |

---

## 1. Project Title and Summary

This project presents an automated computer vision system designed to continuously monitor industrial conveyor belts and warehouse loading bays via security camera streams, detecting and tallying packages and sacks in real time. The software is primarily built for warehouse managers, logistics supervisors, and freight operators who require continuous, hands-off inventory audits during rapid stock loading and dispatch operations. Furthermore, intelligent temporal tracking allows the application to assign a unique persistent identity to each item, thereby following individual packages across successive video frames and preventing double-counting when workers or machinery momentarily obstruct the camera's line of sight. Operators interact with a centralized, web-accessible dashboard to observe live annotated camera feeds, inspect real-time item counts, and examine historical throughput analytics across their facilities.

---

## 2. Problem Statement and Motivation

High-throughput distribution warehouses and freight logistics hubs process thousands of packages and deformable sacks per hour across multiple loading conveyor lanes. Logistics supervisors currently rely on manual tally sheets, hand-clickers, and periodic barcode spot checks—practices prone to worker fatigue, counting inaccuracies reaching 8–15%, and severe dispatch bottlenecks. Automating this via brute-force pixel matching or hand-crafted rules is mathematically intractable; a single 1080p camera stream produces over 2 million pixels per frame at 30 frames per second, yielding a visual search space exceeding 10⁸ possible object bounding configurations per second amid severe visual uncertainties including non-rigid sack deformations, fluctuating ambient lighting, and frequent mutual occlusions by forklift operators. Hand-engineered algorithms fail to disambiguate occluded items from newly introduced objects, resulting in catastrophic identity fragmentation and duplicate tallies. Solving this through robust neural perception and appearance-based temporal association eliminates expensive shipping discrepancies, recovers hundreds of lost labor hours weekly, and guarantees transparent real-time inventory reconciliation for high-velocity supply chains.

---

## 3. Problem Formulation

### 3.1 PEAS Description

| Component | Description for your project |
|---|---|
| **Performance measure** | Inventory counting precision and recall (≥ 95%), Multi-Object Tracking Accuracy (MOTA ≥ 85%), minimization of false positives and ID switches (< 5% rate), real-time throughput (≥ 20 FPS per stream), bounded GPU VRAM (≤ 250 MB per stream, ≤ 1 GB total), and zero buffer-overflow drops under active streaming. |
| **Environment** | Industrial warehouse loading bays and conveyor transit zones monitored by IP cameras via RTSP video feeds; non-stationary background with variable illumination, moving personnel, machinery vibrations, lens distortion, and transient object occlusions. |
| **Actuators** | Real-time bounding-box and trajectory visualizer, WebSocket live frame and telemetry broadcaster (Base64 JPEG stream to frontend), SQLite relational commit engine for finalized audit records, and backend GPU concurrency scheduler. |
| **Sensors** | Networked RTSP IP camera digital video streams (H.264/H.265 frames captured at 25–30 FPS), system resource monitors (sampling host RAM, CPU%, NVIDIA GPU VRAM, and GPU compute utilization). |

### 3.2 Environment and Agent Type

| Property | Classification — tick one |
|---|---|
| Observability | Fully observable ☐ Partially observable ☑ |
| Determinism | Deterministic ☐ Stochastic ☑ |
| Change over time | Static ☐ Dynamic ☑ |
| State and actions | Discrete ☐ Continuous ☑ |
| Agents | Single-agent ☑ Multi-agent / adversarial ☐ |
| Agent type | Simple reflex ☐ Model-based ☑ Goal-based ☐ Utility-based ☐ Learning ☑ |

It is a Model-based Learning Agent as it uses a deep convolutional neural model for perception, coupled with Kalman filter kinematic state models and track history buffers to reason about partially occluded objects over time.

### 3.3 Problem Representation

| Element | Your representation |
|---|---|
| **State representation** | Continuous visual input I_t ∈ ℝ^(H×W×3) and discrete internal tracking state T_t = {(id_i, b_i, v_i, c_i, h_i, f_i)} where id_i is the persistent track ID, b_i = (x, y, w, h) is the 2D bounding box, v_i is the Kalman velocity vector, c_i ∈ {box, sack} is the predicted class, h_i ∈ ℕ is the consecutive hit count, and f_i ∈ ℝ^128 is the visual ReID appearance feature embedding. |
| **Actions / moves** | (1) Infer bounding boxes and class probabilities on I_t; (2) Match detections to active tracks via bipartite cost minimization (spatial IoU + appearance cosine distance); (3) Update Kalman kinematics and increment hit counts h_i; (4) Confirm inventory item if h_i ≥ threshold and id_i ∉ C; (5) Maintain lost tracks up to 90-frame buffer for re-identification; (6) Prune expired tracks. |
| **Initial state** | First video frame I_0 at RTSP initialization t = 0, empty confirmed set C = ∅, and zero counters: count_box = 0, count_sack = 0. |
| **Goal or terminal test** | Termination signal received from user (or end of stream), with invariant satisfied: every unique package and sack observed has been assigned a single persistent track ID, confirmed exactly once upon meeting the stability threshold, and logged without omission or duplicate. |
| **Cost, utility, fitness or evaluation** | Multi-objective utility: U = IDF1 − α(ID Switches) − β\|Count_true − Count_pred\| − γ·max(0, Latency − Latency_target), balancing identification stability, counting error, and real-time constraints. |

---

## 4. Proposed AI Approach

| # | Technique | Role in the project | Own code or library |
|---|---|---|---|
| 1 | YOLOv8n Deep Convolutional Detector | High-speed object localization, bounding-box regression, and classification ('box', 'sack') on incoming video frames. | Library (ultralytics) + custom transfer-learned weights (best.pt) trained on 2,581 industrial domain images. |
| 2 | BoT-SORT Multi-Object Tracker with ReID | Temporal target association, camera motion compensation (GMC via sparse optical flow), Kalman trajectory estimation, and appearance embedding re-identification. | Library (ultralytics BoT-SORT) with fine-tuned config (botsort_accurate.yaml). |
| 3 | Temporal Confirmation Filter (Hysteresis Gate) | Stateful filter to eliminate transient false positives: registers items into the permanent tally only after ≥ min_hits (15) consecutive frames. | Own code (worker.py / count_objects.py). |
| B | Baseline: Naive YOLOv8n + Centroid Accumulator | Single-frame detection accumulating counts via Euclidean centroid proximity, no appearance features or Kalman tracking. | Own code baseline comparison script. |

The selected techniques directly resolve the high dimensionality and partial observability of warehouse video feeds. YOLOv8n serves as an efficient perception engine, extracting object spatial priors in under 12 ms per frame on modest GPU hardware. BoT-SORT pairs these detections with an enhanced Kalman filter to predict target bounding kinematics while compensating for camera jitter via sparse optical flow. When workers occlude packages, BoT-SORT leverages ReID feature cosine similarity to associate recovered tracks with their historical trajectories rather than instantiating spurious new tracks. Our custom temporal confirmation heuristic enforces a strict hysteresis criterion: a track is registered as an official inventory tally only after at least 15 consecutive detection hits, ensuring exceptional counting precision under noisy conditions.

### 4.1 Planned Class Design

| Class | Main Attributes | Main Methods | Relationship |
|---|---|---|---|
| **BaseCounter** | class_names | update(), reset(), counts (abstract property), total() | Abstract base class; `BayesianObjectCounter` inherits from it and overrides all abstract methods. |
| **BayesianObjectCounter** | class_names, p_arrive, p_leave, lambda_present, lambda_noise, confirm_threshold, _belief, _counts, _in_object | update(), reset(), _predict_step(), _update_step(), _poisson_pmf() | Inherits `BaseCounter`; implements two-state HMM Bayesian filter as the second AI technique. |
| **DetectionEnvironment** | source, seed, frame_count, fps, width, height | __iter__(), __next__(), release(), __enter__(), __exit__() | Wraps cv2.VideoCapture; delivers frames to SackDetectorAgent one at a time via iterator protocol. |
| **SackDetectorAgent** | weights_path, conf, tracker, min_hits, device, _model, _hit_counts, _confirmed_ids, _hc_counts, _bayes | perceive(), reset(), hit_counter_counts (property), bayesian_counts (property) | Model-based reflex agent; owns both the YOLO model and a BayesianObjectCounter; called per frame by the detection workers. |

---

## 5. Scope

### In scope

The features you commit to delivering. The following features must work for the project to be considered complete.

| # | Feature | Why it is essential |
|---|---|---|
| 1 | Multi-Stream RTSP Video Ingestion & Decoding | Ingests real-time digital video feeds directly from IP security cameras, handling network reconnects and decoding frames for the vision pipeline. |
| 2 | Deep Object Detection & BoT-SORT Tracking with ReID | Detects 'box' and 'sack' objects in real time and maintains consistent track IDs across camera shake, motion, and temporary line-of-sight occlusions. |
| 3 | Stateful Hysteresis-Based Confirmation Counting | Eliminates false-positive detections and counting fluctuation by requiring consecutive frame hits before committing an item to confirmed inventory. |
| 4 | Real-Time WebSocket Streaming & Web Dashboard | Delivers annotated video frames and live inventory tally updates to operators through an intuitive browser interface. |
| 5 | Hardware Concurrency Governor & Persistent Audit History | Enforces strict dual-stream GPU VRAM safety limits on edge hardware, while archiving run metrics and annotated video playback in SQLite. |

### Out of scope

Deliberately excluded. Naming what you will not build is as important as naming what you will; it shows that your scope is realistic for a group of two.

- **Motorized Camera Control:** Physical Pan-Tilt-Zoom (PTZ) robotics, auto-tracking gimbals, and hardware steering.
- **Text / Barcode Recognition:** OCR for shipping barcodes, QR codes, or serial numbers on packaging.
- **Global Multi-Facility Re-ID:** Tracking packages across distinct warehouses or non-overlapping camera sites.
- **Native Mobile Apps:** Native iOS/Android applications (system is a responsive web dashboard).
- **Multi-Node Cloud Clusters:** Kubernetes or distributed orchestration; solution targets a single GPU edge workstation.

---

## 6. Data or Environment Source

| Field | Detail |
|---|---|
| **Source name and URL** | Industrial Box & Sack Warehouse Logistics Dataset (YOLOv8-annotated) & Recorded RTSP Stream Testbed (data/recordings/) |
| **Type** | Public dataset ☐ Web API ☐ Self-collected ☑ Generated / simulated ☐ |
| **Access requirements** | No key needed ☑ Free key required ☐ (Paid sources are not permitted) |
| **Approximate size** | 3,581 annotated high-resolution training/validation images (~2.4 GB); 4 benchmark evaluation video streams (~850 MB total, 1080p @ 30 fps). |
| **Licence / terms of use** | Academic Research and Educational Use |
| **Offline fallback** | All model weights (weights/best.pt), tracker configuration (backend/botsort_accurate.yaml), and benchmark RTSP video streams (data/recordings/) are stored locally on disk; the containerized system operates fully offline without internet connectivity. |

---

## 7. Evaluation Plan

| Element | Plan |
|---|---|
| **What will be compared** | Proposed YOLOv8n + BoT-SORT (with ReID and Temporal Confirmation Filter) vs. Baseline (pure YOLOv8n frame-by-frame detection count) and Alternative Configuration (YOLOv8n + ByteTrack without ReID). |
| **Metric(s)** | Counting Accuracy (1 − \|True − Pred\| / True), Multi-Object Tracking Accuracy (MOTA), Identification F1-Score (IDF1), ID Switch Count, Frame Processing Throughput (FPS), GPU VRAM Utilization (MB). |
| **Baseline** | Naive frame-by-frame YOLOv8n detection with spatial centroid thresholding (accumulates detection occurrences without appearance tracking or temporal hit confirmation). |
| **Test instances / test set** | 4 standardized benchmark video sequences (20 minutes total footage) recorded in industrial warehouse environments with heavy occlusions, lighting shifts, and high package density. |
| **Number of runs and seed** | 5 independent evaluation runs per test sequence using fixed random seeds (42, 101, 777, 1337, 2026) to ensure deterministic comparison and report mean ± standard deviation. |
| **Planned charts (at least three)** | 1. Grouped bar chart: Counting Accuracy (%) across Baseline, ByteTrack, and BoT-SORT under varying occlusion levels. 2. Line chart: Cumulative ID Switches and False Positive Tracks over time during conveyor clumping. 3. Scatter plot: Inference Throughput (FPS) vs. GPU VRAM Consumption (MB) across single and dual concurrent streams. |

---

## 8. Division of Work

| Member | Modules / components owned | Approximate share |
|---|---|---|
| Muhammad Hassan | YOLOv8n model integration, BoT-SORT tracking with ReID optimization, temporal hit confirmation logic, and hardware concurrency checking. | 40% |
| Muhammad Ali | SQLite database schemas and persistence layer, API endpoints, experimental benchmark evaluation suite, backend multiprocessing worker management | 40% |
| Asad Chotani | React frontend dashboard development and experimental benchmark evaluation suite, real-time WebSocket video and counts streaming. | 20% |

---

## 9. Milestones and Timeline

| # | Milestone — what will be demonstrably working | Target date | Owner |
|---|---|---|---|
| 1 | YOLOv8n model loaded, inferring on sample video feeds with bounding boxes, and validating baseline detection accuracy on static test set. | Week 10 (Nov 16, 2026) | All members |
| 2 | BoT-SORT tracker integrated with hit-counter confirmation; counts verified against manual ground truth on test video. | Week 11 (Nov 23, 2026) | Muhammad Hassan |
| 3 | FastAPI backend complete — all REST endpoints, WebSocket streaming, and SQLite persistence working end-to-end. | Week 11 (Nov 23, 2026) | Muhammad Ali |
| 4 | React frontend complete — video upload page, live frame viewer, dashboard, history, and camera management pages. | Week 12 (Nov 30, 2026) | Asad Chotani |
| 5 | Bayesian HMM layer implemented; full experiment suite run on all 3 configurations; report and repo finalised. | Week 14 (Dec 7, 2026) | Muhammad Hassan |

---

## 10. Expected Outcome and Risks

A working deployment presents an interactive web dashboard allowing operators to register RTSP streams, monitor live annotated video with real-time confirmed counts, and review historical audit logs. Experimentally, BoT-SORT with ReID is expected to achieve over 95% counting accuracy and suppress identity switches by more than 70% compared to naive frame-by-frame detection, sustaining over 25 FPS within a 250 MB VRAM footprint. Two key risks could derail the project: first, GPU memory exhaustion under concurrent camera workloads, mitigated by enforcing a strict concurrency governor capping active pipelines to two with FP16 precision; second, tracking ID fragmentation during prolonged occlusions, mitigated by optimizing ReID distance thresholds and temporal hit hysteresis.

---

## For Office Use — Lab Instructor

| Field | Entry |
|---|---|
| **Proposal status** | Approved ☐ Approved with changes ☐ Revision required ☐ |
| **Comments** | |
| **Signature and date** | |

---

*End of Project Proposal*
