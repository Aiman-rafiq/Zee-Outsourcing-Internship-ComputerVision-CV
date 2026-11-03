
import cv2
import os
import time
from ultralytics import YOLO

# Create output directory
os.makedirs("outputs/predictions", exist_ok=True)

# Video path
video_path = "videos/input.mp4"

# Load YOLO11x
print("Loading YOLO11x...")
model = YOLO("yolo11x.pt")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

# Video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_path = "outputs/predictions/yolo11x_original.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Cannot create output video!")
    cap.release()
    exit()

# Statistics
total_frames = 0
total_person_detections = 0
confidence_scores = []

start_time = time.time()

print("Processing video with YOLO11x...")

# Process frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, imgsz=640, verbose=False)

    result = results[0]

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            confidence = float(box.conf[0])

            total_person_detections += 1
            confidence_scores.append(confidence)

    annotated_frame = result.plot()

    out.write(annotated_frame)

    total_frames += 1

# Release resources
cap.release()
out.release()

# Calculate statistics
processing_time = time.time() - start_time

average_confidence = (
    sum(confidence_scores) / len(confidence_scores)
    if confidence_scores else 0
)

print("\n----- YOLO11x RESULTS -----")
print("Total frames:", total_frames)
print("Total person detections:", total_person_detections)
print("Average confidence:", round(average_confidence, 3))
print("Processing time:", round(processing_time, 2), "seconds")
print("Output saved:", output_path)
