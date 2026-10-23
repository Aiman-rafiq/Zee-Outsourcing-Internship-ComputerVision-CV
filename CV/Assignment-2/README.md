# Week 2 — YOLO Person Detection

## Project Overview
This project uses a pretrained YOLO11 model to detect people in a video using Python and OpenCV.

## Features
- Detects people only (COCO class 0)
- Draws bounding boxes around detected people
- Displays confidence scores
- Processes video frame by frame
- Saves the processed video as `output.mp4`

## Technologies Used
- Python
- Ultralytics YOLO11
- OpenCV

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Place `input.mp4` in the project folder.
3. Run `python main.py`
4. Open `output.mp4` to view the results.

## Output
The processed video contains green bounding boxes and confidence scores around detected people.