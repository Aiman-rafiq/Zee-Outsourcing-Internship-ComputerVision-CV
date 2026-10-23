
import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open input video
video = cv2.VideoCapture("input.mp4")

if not video.isOpened():
    raise FileNotFoundError("Unable to open input.mp4")

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Save processed video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

if not output.isOpened():
    video.release()
    raise RuntimeError("Unable to create output.mp4")

frame_count = 0

# Process the video frame by frame
try:
    while True:
        success, frame = video.read()

        if not success:
            break

        # Detect people only (COCO class 0)
        results = model(frame, classes=[0], conf=0.5, verbose=False)

        # Draw bounding boxes and confidence scores
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            # Draw rectangle
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            # Display label and confidence
            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame, label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

        # Write processed frame to output
        output.write(frame)
        frame_count += 1

finally:
    video.release()
    output.release()

print(f"Processed frames: {frame_count}")
print("Person detection completed!")
print("Output saved as output.mp4")
