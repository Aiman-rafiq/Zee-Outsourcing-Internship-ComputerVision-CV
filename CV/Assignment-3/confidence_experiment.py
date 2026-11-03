
import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO

# 1. Create output directory
os.makedirs("experiments", exist_ok=True)

# 2. Configuration
video_path = "videos/input.mp4"
thresholds = [0.2, 0.4, 0.6, 0.8]

# 3. Load YOLO11n
print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

# 4. Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

# 5. Initialize statistics
results_data = []

total_frames = 0
start_time = time.time()

print("Starting confidence experiments...")

# 6. Process video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO at the lowest threshold
    predictions = model(frame, conf=0.2, verbose=False)

    result = predictions[0]

    # Collect person confidence scores
    person_scores = []

    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id == 0:
            confidence = float(box.conf[0])
            person_scores.append(confidence)

    # Apply all four thresholds
    for threshold in thresholds:

        accepted = [
            score for score in person_scores
            if score >= threshold
        ]

        # Find existing result or create it
        row = next(
            (r for r in results_data
             if r["threshold"] == threshold),
            None
        )

        if row is None:
            row = {
                "threshold": threshold,
                "total_detections": 0,
                "confidence_sum": 0
            }
            results_data.append(row)

        row["total_detections"] += len(accepted)
        row["confidence_sum"] += sum(accepted)

    total_frames += 1

# 7. Release video
cap.release()

processing_time = time.time() - start_time

# 8. Calculate final results
for row in results_data:

    count = row["total_detections"]

    row["average_confidence"] = (
        row["confidence_sum"] / count
        if count > 0 else 0
    )

    row["average_confidence"] = round(
        row["average_confidence"], 3
    )

    del row["confidence_sum"]

# 9. Save results
df = pd.DataFrame(results_data)

df.to_csv(
    "experiments/confidence_results.csv",
    index=False
)

# 10. Display results
print("\n----- CONFIDENCE EXPERIMENT RESULTS -----")

print(df.to_string(index=False))

print("\nTotal frames:", total_frames)
print("Processing time:", round(processing_time, 2), "seconds")
print("Results saved: experiments/confidence_results.csv")
