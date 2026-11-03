
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
df = pd.read_csv(
    "experiments/video_conditions_results.csv"
)

# Create charts folder
os.makedirs("outputs/charts", exist_ok=True)

# Graph 1: Person detections
plt.figure(figsize=(9, 5))

plt.bar(
    df["condition"],
    df["person_detections"]
)

plt.title("Person Detections Under Different Conditions")
plt.xlabel("Video Condition")
plt.ylabel("Total Person Detections")
plt.grid(axis="y")

plt.savefig(
    "outputs/charts/video_condition_detections.png"
)

plt.show()

# Graph 2: Average confidence
plt.figure(figsize=(9, 5))

plt.bar(
    df["condition"],
    df["average_confidence"]
)

plt.title("Average Confidence Under Different Conditions")
plt.xlabel("Video Condition")
plt.ylabel("Average Confidence")
plt.grid(axis="y")

plt.savefig(
    "outputs/charts/video_condition_confidence.png"
)

plt.show()

print("Video condition charts saved successfully!")
