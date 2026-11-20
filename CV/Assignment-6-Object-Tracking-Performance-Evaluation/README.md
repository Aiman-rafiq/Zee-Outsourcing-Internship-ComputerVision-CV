# Assignment 6: Object Tracking Performance Evaluation

## Overview
This project evaluates the performance of YOLO-based person detection and tracking for people counting. The experiments compare different model and tracking configurations using the same input video.

## Dataset / Video
- **Video:** `people.mp4`
- **Resolution:** 1920 × 1080
- **Original FPS:** 25
- **Total Frames:** 617
- **Manual IN Count:** 18
- **Manual OUT Count:** 10

## Baseline Configuration
- Model: YOLO11n
- Confidence Threshold: 0.35
- Tracker: ByteTrack
- Device: CPU

## Experiments

Three experiments were performed, changing one setting at a time.

| Configuration | Total Count Error | Error Percentage | Processing FPS |
|---|---:|---:|---:|
| Baseline: YOLO11n, confidence 0.35 | 7 | 25.00% | 11.34 |
| Experiment 1: Confidence 0.50 | 6 | 21.43% | 11.01 |
| Experiment 2: YOLO11s | 7 | 25.00% | 3.67 |
| Experiment 3: ByteTrack track_buffer 60 | 7 | 25.00% | 7.34 |

## Experimental Observations

### Experiment 1: Confidence Threshold
The confidence threshold was increased from 0.35 to 0.50 while keeping the model and tracker unchanged.

- Total count error decreased from 7 to 6.
- Error percentage improved from 25.00% to 21.43%.
- Processing FPS decreased slightly from 11.34 to 11.01.

This configuration produced the best counting accuracy among the tested configurations.

### Experiment 2: Model Change
The model was changed from YOLO11n to YOLO11s while keeping confidence at 0.35.

- Total count error remained 7.
- OUT count became exact, but IN count error increased.
- Processing FPS decreased significantly to 3.67.

The larger model did not improve overall counting accuracy and was considerably slower on CPU.

### Experiment 3: ByteTrack Tuning
The ByteTrack `track_buffer` parameter was increased from 30 to 60.

- Total count error remained 7.
- Automated counts remained unchanged.
- Processing FPS was 7.34.

This change did not improve counting accuracy in this experiment.

## Conclusion

Among the three tested changes, increasing the confidence threshold to 0.50 produced the best result. It reduced the total count error from 7 to 6 while maintaining a processing speed close to the baseline.

Changing to YOLO11s and increasing the ByteTrack track buffer did not improve the total counting accuracy.

## How to Run

1. Clone the repository.
2. Open the Assignment-6 folder.
3. Create and activate a Python virtual environment.
4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Place the input video at:

```text
data/input/people.mp4
```

6. Run the performance evaluation:

```bash
python src/performance_test.py
```

The script reports automated counts, count errors, processing time, and processing FPS.

## Tools and Technologies
- Python
- OpenCV
- Ultralytics YOLO
- Supervision
- ByteTrack
