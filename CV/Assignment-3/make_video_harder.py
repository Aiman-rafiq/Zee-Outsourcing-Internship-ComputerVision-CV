
import cv2
import numpy as np
import os

# 1. Video paths
video_path = "videos/input.mp4"

output_folder = "outputs/processed_videos"
os.makedirs(output_folder, exist_ok=True)

# 2. Open original video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open original video!")
    exit()

# 3. Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 4. Create video writers
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

dark_writer = cv2.VideoWriter(
    f"{output_folder}/dark_video.mp4",
    fourcc, fps, (width, height)
)

blur_writer = cv2.VideoWriter(
    f"{output_folder}/blur_video.mp4",
    fourcc, fps, (width, height)
)

noise_writer = cv2.VideoWriter(
    f"{output_folder}/noisy_video.mp4",
    fourcc, fps, (width, height)
)

# 5. Check video writers
if not all([
    dark_writer.isOpened(),
    blur_writer.isOpened(),
    noise_writer.isOpened()
]):
    print("Error: Could not create output videos!")
    cap.release()
    exit()

total_frames = 0

print("Creating modified videos...")

# 6. Process each frame
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # A. Darken video
    dark_frame = (frame * 0.4).astype(np.uint8)

    # B. Blur video
    blur_frame = cv2.GaussianBlur(
        frame, (15, 15), 0
    )

    # C. Add random noise
    noise = np.random.normal(
        0, 25, frame.shape
    )

    noisy_frame = np.clip(
        frame.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    # 7. Save modified frames
    dark_writer.write(dark_frame)
    blur_writer.write(blur_frame)
    noise_writer.write(noisy_frame)

    total_frames += 1

# 8. Release resources
cap.release()

dark_writer.release()
blur_writer.release()
noise_writer.release()

print("\n----- VIDEO TRANSFORMATION RESULTS -----")
print("Total frames processed:", total_frames)
print("Dark video saved successfully!")
print("Blur video saved successfully!")
print("Noisy video saved successfully!")
print("Output folder:", output_folder)
