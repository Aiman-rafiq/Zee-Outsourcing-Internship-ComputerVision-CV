# Assignment 4 — Object Tracking

## Overview
This assignment implements a multi-object tracking system using YOLO11n and ByteTrack. The system detects people in a video, assigns tracking IDs, and maintains their identities across consecutive frames.

## Technologies Used
- Python
- YOLO11n
- ByteTrack
- Ultralytics
- OpenCV
- Pandas

## Project Structure
```text
Assignment-4/
├── videos/
├── outputs/
│   └── predictions/
│       ├── tracking_log.csv
│       └── tracking_output.avi
├── main.py
├── custom_bytetrack.yaml
├── yolo11n.pt
└── README.md
```

## Tasks Completed

### Task 1: Object Tracking
- Integrated ByteTrack with the YOLO11n detector.
- Displayed bounding boxes and tracking IDs on detected people.
- Generated an annotated output video.

### Task 2: Unique People Counting
- Performed a manual count of distinct people.
- Extracted unique tracking IDs from the tracking log.
- Compared manual and tracking-based counts.

### Task 3: Tracking Identity Analysis
- Identified a track fragmentation example.
- Observed the same person receiving ID 8 initially and ID 19 later.

## Results
| Measurement | Result |
|---|---:|
| Video frames processed | 256 |
| Manual people count | 7 |
| Unique tracking IDs | 13 |
| Track fragmentation | Observed |

## Output Files
- `tracking_output.avi`: Annotated video with tracking IDs.
- `tracking_log.csv`: Frame-wise tracking information, including IDs, confidence scores, and bounding-box coordinates.

## How to Run

1. Install the required dependencies:
   ```bash
   pip install ultralytics opencv-python pandas
   ```

2. Run the tracking script:
   ```bash
   python main.py
   ```

3. Check the generated video and CSV log in `outputs/predictions/`.
