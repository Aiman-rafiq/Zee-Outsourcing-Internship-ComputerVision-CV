
import cv2
from ultralytics import YOLO

# Step 1: Load YOLO model
model = YOLO("yolo11n.pt")

# Step 2: Open input video
video_path = "data/input/people.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Step 3: Process video frames
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Step 4: Detect people using YOLO
    results = model(
        frame,
        classes=[0],
        conf=0.35,
        device="cpu",
        verbose=False
    )

    # Step 5: Draw detection results
    annotated_frame = results[0].plot()

    # Step 6: Resize frame for display
    display_frame = cv2.resize(
        annotated_frame,
        (960, 540)
    )

    cv2.imshow("YOLO Person Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 7: Release resources
cap.release()
cv2.destroyAllWindows()
