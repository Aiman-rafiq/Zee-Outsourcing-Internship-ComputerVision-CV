
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "tracking_output.avi"
OUTPUT_CSV = OUTPUT_DIR / "tracking_log.csv"

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = BASE_DIR / "custom_bytetrack.yaml"

CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5


# --------------------------------------------------
# 2. Load YOLO model
# --------------------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------------------------
# 3. Open input video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open input video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nVideo Information")
print("-----------------")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")


# --------------------------------------------------
# 4. Initialize output video writer
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# --------------------------------------------------
# 5. Initialize tracking records
# --------------------------------------------------

unique_track_ids = set()
frame_number = 0

csv_file = open(
    OUTPUT_CSV,
    mode="w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time_seconds",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])


# --------------------------------------------------
# 6. Process video frame by frame
# --------------------------------------------------

print("\nStarting object tracking...\n")

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Run YOLO detection and ByteTrack tracking.
        # Class 0 represents people in the COCO dataset.
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=str(TRACKER_CONFIG),
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        annotated_frame = frame.copy()

        # Process detected objects
        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            if boxes.id is not None:

                track_ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for track_id, box, confidence in zip(
                    track_ids,
                    coordinates,
                    confidences
                ):

                    x1, y1, x2, y2 = map(int, box)

                    unique_track_ids.add(track_id)

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    # Display tracking ID and confidence
                    label = f"Person ID: {track_id} ({confidence:.2f})"

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    # Save tracking information to CSV
                    time_seconds = frame_number / fps if fps > 0 else 0

                    csv_writer.writerow([
                        frame_number,
                        round(time_seconds, 2),
                        track_id,
                        round(float(confidence), 3),
                        x1,
                        y1,
                        x2,
                        y2
                    ])

        # Display frame number
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        # Save annotated frame
        writer.write(annotated_frame)

        # Progress update
        if frame_number % 50 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

finally:
    # Release resources even if processing stops unexpectedly
    cap.release()
    writer.release()
    csv_file.close()


# --------------------------------------------------
# 7. Display final tracking results
# --------------------------------------------------

print("\nTracking completed!")
print("-------------------")

print(f"Total frames processed: {frame_number}")
print(f"Unique tracking IDs: {len(unique_track_ids)}")
print(f"Tracking IDs: {sorted(unique_track_ids)}")

print(f"\nOutput video saved at:\n{OUTPUT_VIDEO}")
print(f"\nTracking CSV saved at:\n{OUTPUT_CSV}")
