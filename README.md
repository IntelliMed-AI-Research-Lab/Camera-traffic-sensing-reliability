# Right Count, Wrong Reasons: Error Cancellation and the Real Drivers of Failure in Detector-Based Vehicle Counting for Camera-Based Non-Invasive Video Traffic Sensing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/right-count-wrong-reasons/blob/main/Traffic_Density_Project_v4_Final.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/detector-YOLOv8-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A pre-registered reliability study of camera-based, non-invasive traffic sensing. It asks **when and why detector-derived vehicle counts fail**, and **how much traffic "vital signs" (density, flow, speed) change** when only the detector's operating point changes.

> Everything lives in one notebook, [`Traffic_Density_Project_v4_Final.ipynb`](Traffic_Density_Project_v4_Final.ipynb), with all outputs saved. Open it in Google Colab, choose a **T4 GPU**, and press **Run all**. Nothing to upload.

---

## Motivation

A roadside camera is a **non-invasive** traffic sensor: no road works, no contact with vehicles, and one device covers many lanes. Turning video into traffic measurements needs an object detector, and detectors make errors.

The usual way to judge a counter is to compare the **total count** against the truth. That can be misleading. A detector can miss many vehicles and also hallucinate many vehicles, and the two errors cancel. The total looks perfect while the detector is wrong on individual vehicles. This project measures that effect and asks what is really behind the failures.

## Key findings

| # | Finding | Evidence (from the notebook) |
|---|---|---|
| 1 | **Right count, wrong reasons.** Net count bias can reach ~0% while recall and precision are both far from perfect. | Highway, conf 0.6: predicted 937 = truth 937 (bias 0.0%), yet recall = precision = 0.912. VisDrone fine-tuned, conf 0.25: bias −0.4%, yet recall 0.812 / precision 0.815. |
| 2 | **Count bias flips sign with the confidence threshold.** | Highway: +18.4% (conf 0.25) to 0.0% (conf 0.6). Fine-tuned VisDrone: −0.4% (0.25) to −28.7% (0.5). |
| 3 | **Off-the-shelf detectors badly undercount aerial scenes.** | COCO YOLOv8n/s/m on VisDrone: recall 0.30–0.56, net bias −32% to −68% at conf 0.25–0.5. |
| 4 | **Object size and occlusion are the main drivers of missed vehicles.** | Odds of detection rise ~3.6–5.6× per unit of log-size and fall to ~0.35–0.55× under occlusion (all p < 0.001). |
| 5 | **Density per se is not the driver for the fine-tuned detector** (pre-registered test). | Standardised sparse-vs-dense recall drop: −0.013 [−0.046, +0.019]. Verdict: not supported. |
| 6 | **A raw density effect is mostly a small-object effect.** | Raw drop 0.175–0.270 for all vehicles, but only 0.031–0.037 for ≥32 px vehicles in YOLOv8n/s (CI includes 0). YOLOv8m keeps a residual 0.100. |
| 7 | **Traffic "vital signs" are not equally robust to detector settings.** | Across 12 settings on one video: density range 57%, speed 17%, flow 10% (of the reference value). |
| 8 | **Even a "well-calibrated" total hides per-image error.** | VisDrone YOLOv8n at its oracle threshold (conf 0.10): bias +4.1%, yet normalised MAE is still 0.345. |

> Numbers come straight from the executed notebook. Findings 6 (COCO rows) and the mechanism grid were designed after seeing earlier results and are labelled **exploratory** in the notebook.

### Figures

**Figure 1: qualitative results** (green = ground truth found, blue = correct detection, red = missed, orange dashed = false positive)

![Figure 1](assets/fig1_qualitative.png)

**Figure 2: quantitative results** (recall vs density, size, occlusion; bias flip; nMAE; mechanism test)

![Figure 2](assets/fig2_results.png)

**Figure 3: traffic vital signs and their sensitivity to detector settings**

![Figure 3](assets/fig3_vital_signs.png)

---

## Studies

| Study | Question | Data | Result |
|---|---|---|---|
| **A** | Does recall fall as scenes get denser? | 90 validation images, top-view highway, 937 vehicles, fine-tuned YOLOv8n | **Not supported.** Recall drop −0.031 [−0.089, +0.019]. |
| **B** | Same question with off-the-shelf detectors | VisDrone val, 548 images, 17,040 vehicles, COCO YOLOv8n/s/m | **Partial.** Drop appears for all vehicles but mostly vanishes for ≥32 px. |
| **C** | Is there a density effect beyond size and occlusion? | VisDrone, YOLOv8n fine-tuned on 6,471 train images (35 epochs, 960 px) | **Not supported** (pre-registered). Standardised drop −0.013 [−0.046, +0.019]. |
| **E** | How do MAE/RMSE/bias behave? Does NMS IoU or input size explain the residual density effect? | VisDrone val | Threshold controls bias. NMS IoU changes nothing (ratio 0.99–1.01). Input size has a modest effect. Remaining effect fits a **domain gap** (exploratory). |
| **F** | How stable are density, flow and speed under different detector settings? | One sample highway video, 601 frames, 12 tracking runs (conf × input size) | Density is unstable (57%), speed moderate (17%), flow stable (10%). Stable does **not** mean accurate: there is no video ground truth. |

### How the standardised density test works
Vehicles are grouped into 12 strata (4 size bands × 3 annotated occlusion levels). Recall in the sparsest third of images is compared with the densest third **using the same size-and-occlusion mix in both groups**, with a bootstrap over images. This asks: *does image density matter once vehicle size and occlusion are held fixed?*

**Pre-registered rule (Study C):** a density effect beyond size and occlusion is *supported* only if the 95% bootstrap CI of the standardised drop for the fine-tuned detector lies entirely above zero.

### Study F metric definitions
- **Density**: vehicles visible per frame.
- **Flow**: vehicles per minute crossing a virtual counting line (fixed for all settings).
- **Speed**: median per-track speed in vehicle lengths per second (the video has no camera calibration, so no km/h).

---

## Data

| Dataset | Use | Source and licence |
|---|---|---|
| Top-View Vehicle Detection Image Dataset (F. Nekouei), 536 train / 90 val images, one class, plus a sample video | Highway detector, Study A, Study F | Kaggle, CC BY 4.0 (frames from Pexels videos) |
| VisDrone2019-DET (Tianjin University) | Studies B, C, E | Check the VisDrone licence before redistributing or publishing |

Data is downloaded automatically by the notebook and is **not** stored in this repository.

## How to run

**Option 1: Google Colab (recommended)**
1. Click the *Open In Colab* badge at the top.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**. Keep the browser tab open.

**Time budget on a Colab T4**

| Step | Approx. time |
|---|---|
| Highway detector training (100 epochs) | 13 min (measured) |
| Figures, tracking, Studies A and B | 15–25 min |
| VisDrone fine-tuning (35 epochs, 960 px) | ~2.1 h |
| Study E (inference only) | 15–30 min |
| Study F (12 tracking runs) | 5–10 min |
| **Total** | **~3.1–3.7 h** |

To stay under a 3.5 h limit, set `FT_MAX_HOURS = 1.5` or `FT_IMGSZ = 800` in the settings cell. Set `SAVE_TO_DRIVE = True` so a disconnect can be resumed.

**Option 2: local**
```bash
git clone https://github.com/YOUR_USERNAME/right-count-wrong-reasons.git
cd right-count-wrong-reasons
pip install -r requirements.txt
jupyter notebook Traffic_Density_Project_v4_Final.ipynb
```
A CUDA GPU is strongly recommended. Without one, the notebook falls back to 15 epochs, a single small model, and skips VisDrone fine-tuning.

## Repository structure

```
.
├── README.md
├── Traffic_Density_Project_v4_Final.ipynb   # full study, outputs included
├── requirements.txt
├── LICENSE
├── .gitignore
└── assets/
    ├── fig1_qualitative.png
    ├── fig2_results.png
    └── fig3_vital_signs.png
```

Running the notebook also writes `paper_assets/` (vector PDFs, LaTeX tables) and `project_results.zip` (figures, CSV tables). These are generated outputs and are ignored by git.

## Limitations

- **Single video for Study F, and no ground truth for flow or speed.** Sensitivity is measured, accuracy is not.
- **Model selection on the validation split.** The best checkpoint is picked on the same split that is analysed, so absolute numbers are slightly optimistic. The study focuses on relative effects.
- **Small model.** Fine-tuning uses YOLOv8n only. A larger detector and a second benchmark with per-object occlusion labels (e.g. UA-DETRAC, UAVDT) would strengthen the conclusions.
- **Mechanism grid is exploratory**, and its domain-gap explanation is a hypothesis, not a proven cause.
- **No comparison with inductive loops.** No loop data was available, so no claim is made against in-road sensors.

## Citation

If you use this work, please cite:

```bibtex
@misc{rightcount2026,
  title  = {Right Count, Wrong Reasons: Error Cancellation and the Real Drivers of Failure in Detector-Based Vehicle Counting for Camera-Based Non-Invasive Video Traffic Sensing},
  author = {YOUR NAME},
  year   = {2026},
  howpublished = {\url{https://github.com/YOUR_USERNAME/right-count-wrong-reasons}}
}
```

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for detection and tracking (ByteTrack).
- F. Nekouei, *Top-View Vehicle Detection Image Dataset* (CC BY 4.0).
- Tianjin University, *VisDrone2019-DET*.

## License

Code and notebook are released under the MIT License (see [`LICENSE`](LICENSE)). Datasets keep their own licences.
