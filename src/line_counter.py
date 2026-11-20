
import cv2
import supervision as sv
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO("yolo11n.pt")

# 2. Open input video
cap = cv2.VideoCapture("data/input/people.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 3. Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 4. Define virtual line
start = sv.Point(0, height // 2)
end = sv.Point(width, height // 2)

line_zone = sv.LineZone(
    start=start,
    end=end,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# 5. Process video
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # 6. Detect and track people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.35,
        tracker="bytetrack.yaml",
        device="cpu",
        verbose=False
    )

    # 7. Convert YOLO detections to Supervision format
    detections = sv.Detections.from_ultralytics(results[0])

    # 8. Check line crossings
    line_zone.trigger(detections)

    # 9. Draw detection boxes
    annotated_frame = results[0].plot()

    # 10. Draw virtual line
    cv2.line(
        annotated_frame,
        (0, height // 2),
        (width, height // 2),
        (0, 255, 255),
        3
    )

    # 11. Display entry and exit counts
    cv2.putText(
        annotated_frame,
        f"IN: {line_zone.in_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        annotated_frame,
        f"OUT: {line_zone.out_count}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    # 12. Display frame
    display_frame = cv2.resize(annotated_frame, (960, 540))

    cv2.imshow("Entry Exit Counter", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 13. Final counts
print("\nFinal Counting Results")
print("----------------------")
print("Total IN:", line_zone.in_count)
print("Total OUT:", line_zone.out_count)

cap.release()
cv2.destroyAllWindows()
