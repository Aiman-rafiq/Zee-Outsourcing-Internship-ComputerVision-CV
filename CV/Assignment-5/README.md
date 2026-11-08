# Week 5: From Tracks to Events
## Pedestrian Entry/Exit Counter and Restricted Zone Alert System

### Project Overview

This project implements a computer vision-based pedestrian monitoring system using YOLO, ByteTrack, OpenCV, and Supervision.

The system detects and tracks pedestrians in a video, counts their entry and exit events using a virtual line, and generates alerts when people enter a predefined restricted zone.

### Objectives

- Detect pedestrians using YOLO.
- Track pedestrians across video frames using ByteTrack.
- Assign persistent tracking IDs to detected people.
- Count pedestrian entries and exits using a virtual line.
- Detect people entering a restricted polygon zone.
- Generate alerts for restricted zone entries.
- Compare automated counting results with manual observations.

### Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video processing and visualization |
| YOLO11n | Pedestrian detection |
| ByteTrack | Multi-object tracking |
| Supervision | Line crossing and polygon zone detection |
| NumPy | Coordinate and array operations |

### Project Structure

```text
Assignment-5/
│
├── data/
│   ├── input/
│   └── output/
│
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── line_counter.py
│   ├── zone_alert.py
│   └── main.py
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Implementation Details

#### 1. Person Detection

YOLO11n is used to detect pedestrians in the input video. The system identifies people and draws bounding boxes around detected objects.

#### 2. Object Tracking

ByteTrack maintains pedestrian identities across consecutive frames. Each detected person receives a tracking ID that helps identify their movement.

#### 3. Entry and Exit Counting

A virtual horizontal line is placed in the video.

The Supervision LineZone functionality detects when tracked pedestrians cross the line and maintains separate entry and exit counts.

The bottom-center point of each bounding box is used as the reference point for crossing detection.

#### 4. Restricted Zone Monitoring

A polygon is defined over a selected area in the video.

The system checks whether the bottom-center point of a tracked pedestrian enters the polygon. An alert is generated when a person enters the restricted zone.

### Results and Evaluation

The automated counting system was compared with manual observations.

| Measurement | Automated Count | Manual Count |
|---|---:|---:|
| Entry (IN) | 16 | 18 |
| Exit (OUT) | 15 | 10 |
| Total Crossings | 31 | 28 |

The automated system recorded 31 crossings, while the manual count recorded 28 crossings.

The differences may be related to tracking ID switches, group crossings, line placement, and temporary detection failures.

### Installation and Setup

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd Assignment-5
```

3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

6. Place your input video inside:

```text
data/input/people.mp4
```

### Running the Project

Run individual modules using:

```bash
python src/detector.py
```

```bash
python src/tracker.py
```

```bash
python src/line_counter.py
```

```bash
python src/zone_alert.py
```

### Notes

- The project uses CPU-based inference.
- YOLO model weights are downloaded automatically when required.
- The input video and generated output videos are not included in the repository.
- The project was tested using a 1920 × 1080 pedestrian video at 25 FPS.

### Conclusion

This project demonstrates how object detection and multi-object tracking can be used to convert pedestrian movement into meaningful events.

It implements pedestrian detection, tracking, entry/exit counting, and restricted zone monitoring using computer vision techniques.
