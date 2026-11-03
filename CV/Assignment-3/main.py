
import cv2
from ultralytics import YOLO
import time

# Step 1: Load video and model
video_path = "videos/input.mp4"

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

# Step 2: Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Step 3: Create output video
output_path = "outputs/predictions/yolo11n_original.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Step 4: Initialize statistics
total_frames = 0
total_person_detections = 0
confidence_scores = []
start_time = time.time()

# Step 5: Process video frame by frame
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)

    result = results[0]

    # Count person detections (class 0)
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            confidence = float(box.conf[0])

            total_person_detections += 1
            confidence_scores.append(confidence)

    # Draw bounding boxes
    annotated_frame = result.plot()

    out.write(annotated_frame)

    total_frames += 1

# Step 6: Release resources
cap.release()
out.release()

# Step 7: Calculate statistics
end_time = time.time()
processing_time = end_time - start_time

average_confidence = (
    sum(confidence_scores) / len(confidence_scores)
    if confidence_scores else 0
)

print("\n----- YOLO11n BASELINE RESULTS -----")
print("Total frames:", total_frames)
print("Total person detections:", total_person_detections)
print("Average confidence:", round(average_confidence, 3))
print("Processing time:", round(processing_time, 2), "seconds")
print("Output saved:", output_path)
