
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture("data/input/people.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.35,
        tracker="bytetrack.yaml",
        device="cpu",
        verbose=False
    )

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Display video
    display_frame = cv2.resize(annotated_frame, (960, 540))

    cv2.imshow("YOLO Person Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
