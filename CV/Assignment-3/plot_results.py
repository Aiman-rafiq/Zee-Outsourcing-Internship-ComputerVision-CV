
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load experimental results
df = pd.read_csv("experiments/confidence_results.csv")

# Create charts folder
os.makedirs("outputs/charts", exist_ok=True)

# Graph 1: Threshold vs detections
plt.figure(figsize=(8, 5))

plt.plot(
    df["threshold"],
    df["total_detections"],
    marker="o"
)

plt.title("Confidence Threshold vs Person Detections")
plt.xlabel("Confidence Threshold")
plt.ylabel("Total Person Detections")
plt.grid(True)

plt.savefig("outputs/charts/detection_count.png")
plt.show()


# Graph 2: Threshold vs average confidence
plt.figure(figsize=(8, 5))

plt.plot(
    df["threshold"],
    df["average_confidence"],
    marker="o"
)

plt.title("Confidence Threshold vs Average Confidence")
plt.xlabel("Confidence Threshold")
plt.ylabel("Average Confidence")
plt.grid(True)

plt.savefig("outputs/charts/average_confidence.png")
plt.show()

print("Both charts saved successfully!")
