
import cv2
import numpy as np
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

# 4. Define restricted polygon around benches
polygon = np.array([
    [150, 250],
    [1650, 250],
    [1700, 650],
    [150, 650]
])

# 5. Create polygon zone
zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

# Track people currently inside the zone
previous_inside_ids = set()

# 6. Process video
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # 7. Detect and track people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.35,
        tracker="bytetrack.yaml",
        device="cpu",
        verbose=False
    )

    # 8. Convert detections
    detections = sv.Detections.from_ultralytics(results[0])

    # 9. Check which people are inside the polygon
    inside_mask = zone.trigger(detections)

    current_inside_ids = set()

    if detections.tracker_id is not None:

        for track_id, is_inside in zip(
            detections.tracker_id,
            inside_mask
        ):

            if is_inside and track_id is not None:
                current_inside_ids.add(int(track_id))

    # 10. Generate alert only when someone newly enters
    new_entries = current_inside_ids - previous_inside_ids

    for track_id in new_entries:
        print(f"ALERT: Person ID {track_id} entered restricted zone!")

    previous_inside_ids = current_inside_ids

    # 11. Draw detection boxes
    annotated_frame = results[0].plot()

    # 12. Draw restricted polygon
    cv2.polylines(
        annotated_frame,
        [polygon],
        True,
        (0, 0, 255),
        3
    )

    # 13. Display zone information
    cv2.putText(
        annotated_frame,
        "RESTRICTED ZONE",
        (polygon[0][0], polygon[0][1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"People Inside: {len(current_inside_ids)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # 14. Display video
    display_frame = cv2.resize(
        annotated_frame,
        (960, 540)
    )

    cv2.imshow("Restricted Zone Alert", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
