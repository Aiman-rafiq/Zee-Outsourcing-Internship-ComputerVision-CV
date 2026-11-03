
import cv2
import os
import time
import pandas as pd
from ultralytics import YOLO

# 1. Video paths
videos = {
    "Original": "videos/input.mp4",
    "Dark": "outputs/processed_videos/dark_video.mp4",
    "Blur": "outputs/processed_videos/blur_video.mp4",
    "Noise": "outputs/processed_videos/noisy_video.mp4"
}

# 2. Load YOLO11n
print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

results_data = []

# 3. Evaluate each video
for video_name, video_path in videos.items():

    print(f"\nProcessing {video_name} video...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening {video_name} video!")
        continue

    total_frames = 0
    total_detections = 0
    confidence_scores = []

    start_time = time.time()

    # 4. Process frames
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        predictions = model(
            frame,
            conf=0.25,
            imgsz=640,
            verbose=False
        )

        result = predictions[0]

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id == 0:

                confidence = float(box.conf[0])

                total_detections += 1
                confidence_scores.append(confidence)

        total_frames += 1

    cap.release()

    # 5. Calculate statistics
    processing_time = time.time() - start_time

    average_confidence = (
        sum(confidence_scores) / len(confidence_scores)
        if confidence_scores else 0
    )

    results_data.append({
        "condition": video_name,
        "frames": total_frames,
        "person_detections": total_detections,
        "average_confidence": round(average_confidence, 3),
        "processing_time": round(processing_time, 2)
    })

# 6. Save results
os.makedirs("experiments", exist_ok=True)

df = pd.DataFrame(results_data)

df.to_csv(
    "experiments/video_conditions_results.csv",
    index=False
)

# 7. Display results
print("\n----- VIDEO CONDITIONS RESULTS -----")
print(df.to_string(index=False))

print("\nResults saved successfully!")
print("File: experiments/video_conditions_results.csv")
