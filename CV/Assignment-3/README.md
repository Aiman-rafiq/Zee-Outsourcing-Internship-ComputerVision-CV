# Computer Vision – Assignment 3
## YOLO11 Model Evaluation and Performance Analysis

### 1. Introduction

This assignment focuses on evaluating object detection models using YOLO11. The objective is to understand how different models, confidence thresholds, and video conditions affect object detection performance.

The experiments use the same input video selected for Assignment 2.

### 2. Models Used

- **YOLO11n:** Nano model.
- **YOLO11x:** Extra Large model.

Both models were evaluated on the same original video using a confidence threshold of 0.25.

### 3. Experiments

#### Experiment 1: Model Comparison

Compared YOLO11n and YOLO11x using:
- Total person detections.
- Average confidence.
- Processing time.

| Metric | YOLO11n | YOLO11x |
|---|---:|---:|
| Total Frames | 256 | 256 |
| Person Detections | 2,122 | 2,236 |
| Average Confidence | 0.623 | 0.679 |
| Processing Time | 49.76 sec | 386.51 sec |

**Observation:** YOLO11x produced more person detections and a higher average confidence, but required significantly more processing time.

#### Experiment 2: Confidence Threshold Analysis

Tested four confidence thresholds using YOLO11n:

- 0.2
- 0.4
- 0.6
- 0.8

| Threshold | Person Detections | Average Confidence |
|---|---:|---:|
| 0.2 | 2,365 | 0.582 |
| 0.4 | 1,769 | 0.683 |
| 0.6 | 1,308 | 0.740 |
| 0.8 | 305 | 0.837 |

**Observation:** Increasing the confidence threshold reduced the number of accepted detections while increasing their average confidence.

#### Experiment 3: Video Condition Analysis

Tested YOLO11n on four video conditions:

1. Original video
2. Darkened video
3. Blurred video
4. Noisy video

| Condition | Frames | Person Detections | Average Confidence |
|---|---:|---:|---:|
| Original | 256 | 2,122 | 0.623 |
| Dark | 256 | 2,171 | 0.599 |
| Blur | 256 | 2,150 | 0.610 |
| Noise | 256 | 2,287 | 0.583 |

**Observation:** Modified video conditions changed the detection behavior. The noisy video produced the highest number of detections but the lowest average confidence.

### 4. Visualizations

Matplotlib was used to generate four charts:

- Confidence Threshold vs Average Confidence
- Confidence Threshold vs Person Detections
- Video Conditions vs Person Detections
- Video Conditions vs Average Confidence

The generated PNG charts are included in this repository.

### 5. Technologies Used

- Python
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pandas
- Matplotlib

### 6. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The YOLO model weights are downloaded automatically by Ultralytics when the corresponding scripts are executed, provided an internet connection is available.

### 7. Running the Experiments

Run the Python scripts from the Assignment-3 project directory:

```bash
python main.py
python compare_models.py
python confidence_experiment.py
python make_video_harder.py
python evaluate_video_conditions.py
python plot_results.py
python plot_conditions.py
```

Place the original video at:

```text
videos/input.mp4
```

### 8. Conclusion

The experiments demonstrated that:

- YOLO11x generated more person detections and higher average confidence than YOLO11n on the selected video, but took considerably longer.
- Increasing the confidence threshold reduced the number of accepted detections and increased average confidence.
- Darkening, blurring, and adding noise changed the model's detection behavior.
- Higher confidence and higher detection counts do not necessarily mean higher detection accuracy.

**Note:** These experiments measure detection counts and confidence scores. Precision, recall, and mAP were not calculated because ground-truth annotations were not available.

---

**Assignment:** Computer Vision – Week 3  
**Topic:** Evaluating the Model  
**Repository:** Zee-Outsourcing-ML-Internship
