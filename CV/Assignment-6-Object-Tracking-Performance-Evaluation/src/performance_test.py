
import cv2
import time
from pathlib import Path

from ultralytics import YOLO
import supervision as sv


# ==============================
# EXPERIMENT CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATH = BASE_DIR / "data" / "input" / "people.mp4"
TRACKER_CONFIG = BASE_DIR / "src" / "custom_bytetrack.yaml"

MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.35
DEVICE = "cpu"

# Manual counts from Assignment 5
MANUAL_IN = 18
MANUAL_OUT = 10


# ==============================
# LOAD MODEL
# ==============================

print("Loading YOLO model...")

model = YOLO(MODEL_NAME)

print("Running Experiment 3: ByteTrack tuning...")


# ==============================
# OPEN VIDEO
# ==============================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ==============================
# INITIALIZE LINE COUNTER
# ==============================

line_y = height // 2

line_zone = sv.LineZone(
    start=sv.Point(0, line_y),
    end=sv.Point(width, line_y),
    triggering_anchors=(
        sv.Position.BOTTOM_CENTER,
    )
)


# ==============================
# PROCESS VIDEO
# ==============================

processed_frames = 0

start_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        tracker=str(TRACKER_CONFIG),
        device=DEVICE,
        verbose=False
    )

    if results and results[0].boxes is not None:

        detections = sv.Detections.from_ultralytics(
            results[0]
        )

        line_zone.trigger(detections)

    processed_frames += 1

end_time = time.perf_counter()

cap.release()


# ==============================
# CALCULATE RESULTS
# ==============================

processing_time = end_time - start_time

processing_fps = (
    processed_frames / processing_time
    if processing_time > 0
    else 0
)

automated_in = line_zone.in_count
automated_out = line_zone.out_count

in_error = abs(MANUAL_IN - automated_in)
out_error = abs(MANUAL_OUT - automated_out)

total_error = in_error + out_error

total_manual = MANUAL_IN + MANUAL_OUT

error_percentage = (
    (total_error / total_manual) * 100
    if total_manual > 0
    else 0
)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n========== EXPERIMENT 3 RESULTS ==========")

print(f"Model: {MODEL_NAME}")
print(f"Confidence: {CONFIDENCE}")
print(f"Tracker: ByteTrack (custom configuration)")
print(f"track_buffer: 60")
print(f"Device: {DEVICE}")

print("\nVideo Information")
print(f"Resolution: {width} x {height}")
print(f"Original FPS: {original_fps}")
print(f"Total Frames: {total_frames}")

print("\nAutomated Results")
print(f"IN: {automated_in}")
print(f"OUT: {automated_out}")

print("\nManual Results")
print(f"IN: {MANUAL_IN}")
print(f"OUT: {MANUAL_OUT}")

print("\nCount Error")
print(f"IN Error: {in_error}")
print(f"OUT Error: {out_error}")
print(f"Total Error: {total_error}")
print(f"Error Percentage: {error_percentage:.2f}%")

print("\nPerformance")
print(f"Processed Frames: {processed_frames}")
print(f"Processing Time: {processing_time:.2f} seconds")
print(f"Processing FPS: {processing_fps:.2f}")

print("==========================================")
