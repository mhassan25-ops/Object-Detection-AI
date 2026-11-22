# AL-2002 Artificial Intelligence Laboratory — Project Proposal
**FAST National University of Computer and Emerging Sciences, Karachi Campus**  
**Department of Software Engineering · Fall 2026**

---

| Field | Detail |
|---|---|
| **Project Title** | Sack and Box Detection and Counting System |
| **Course** | AL-2002 Artificial Intelligence (Laboratory) |
| **Group Members** | Muhammad Hassan (24K-3051) · Muhammad Ali (24K-3103) · Asad Chotani (24K-3072) |
| **Repository** | https://github.com/mhassan25-ops/Object-Detection-AI |

---

## 1. Problem Description

Warehouses and loading docks require accurate counts of physical goods — sacks and boxes — as they pass through the facility. Manual counting is slow, error-prone, and impractical at scale. The goal of this project is to build an automated vision-based system that watches a camera feed, detects individual sacks and boxes, and counts each distinct object exactly once regardless of how many frames it appears in.

The core challenge is *counting*, not detection. The same physical object is visible across hundreds of consecutive frames. A naive counter that increments on every detection overcounts by orders of magnitude. The system must therefore distinguish between "this is the same object from last frame" and "this is a new object that just entered the scene".

---

## 2. Problem Formulation

**PEAS:**
- **Performance measure:** Absolute counting error |predicted − ground truth|; noise rejection rate; processing throughput (fps)
- **Environment:** Warehouse video stream — pre-recorded `.mp4` or live RTSP camera feed
- **Actuators:** None (purely perceptual agent)
- **Sensors:** Camera frames as BGR numpy arrays via OpenCV

**Environment classification:** Partially observable, stochastic, sequential, dynamic, continuous, single-agent.

**Agent type:** Model-based reflex agent — maintains internal state (track hit-counts, confirmed IDs, Bayesian belief) and acts based on current percept combined with that model.

---

## 3. Chosen AI Techniques

### Core Technique — YOLOv8 + BoT-SORT with Hit-Counter (Lab 11)

A custom YOLOv8n model trained on warehouse footage detects `box` and `sack` objects in each frame. BoT-SORT assigns persistent track IDs across frames. A hit-counter confirms a track as a unique object only after it has appeared for `min_hits` frames, filtering out spurious short-lived detections.

### Second Technique — Bayesian Two-State HMM (Lab 09)

A two-state Hidden Markov Model models each class independently. The hidden state is `{IN_SCENE, NOT_IN_SCENE}`, observations are raw per-frame detection counts, and emissions follow a Poisson distribution. The forward algorithm updates `P(IN_SCENE)` each frame. Confirm/exit hysteresis prevents the same persistent object from being counted more than once.

Both techniques run on the same video and their counts are compared against a manually counted ground truth.

---

## 4. OOP Design

Three classes following the Labs 01–02 requirement:

| Class | File | Role |
|---|---|---|
| `BaseCounter` | `src/base_counter.py` | Abstract base class — defines the counter interface |
| `BayesianObjectCounter` | `src/bayesian_counter.py` | Inherits `BaseCounter` — implements the HMM technique |
| `DetectionEnvironment` | `src/environment.py` | Wraps video source, delivers frames via iterator protocol |
| `SackDetectorAgent` | `src/agent.py` | Model-based reflex agent owning both counting methods |

Inheritance: `BayesianObjectCounter` extends `BaseCounter`, overriding `update()`, `reset()`, `counts`, and `__repr__()`.

---

## 5. Data and Environment

- **Training data:** Custom-annotated warehouse footage — two classes: `box`, `sack`
- **Model weights:** YOLOv8n fine-tuned — `runs/detect/sack_box_detect/v1/weights/best.pt`
- **Test video:** `video.mp4` — 70 seconds, 848×480, 25 fps
- **Ground truth:** Manually counted — 3 sacks, 6 boxes, 9 total

---

## 6. Evaluation Plan

Three configurations tested on the same video with `random.seed(42)`:

| Config | conf threshold | min\_hits |
|---|---|---|
| A | 0.30 | 3 |
| B | 0.40 | 3 |
| C | 0.55 | 5 |

**Metrics:** Absolute error per class and total; noise tracks discarded; processing speed (fps).  
**Comparison:** Hit-Counter method vs Bayesian HMM layer on same video instances.  
**Charts:** Counts vs ground truth (Fig 1), absolute error (Fig 2), noise vs fps (Fig 3).

---

## 7. Work Division and Milestones

| Member | Responsibility |
|---|---|
| Muhammad Hassan (24K-3051) | AI core — OOP classes, detection workers, Bayesian HMM, experiments |
| Muhammad Ali (24K-3103) | Backend API — FastAPI server, REST endpoints, WebSocket, database |
| Asad Chotani (24K-3072) | Frontend — React UI, video upload page, live frame viewer |

| Milestone | Target |
|---|---|
| OOP classes and Bayesian layer complete | Week 10 |
| Backend API and WebSocket complete | Week 11 |
| Frontend and integration complete | Week 12 |
| Experiments run, results recorded | Week 13 |
| Report written and repo finalised | Week 14 |
| Submission | Week 15 |

---

*End of Proposal — AL-2002 · Fall 2026*
