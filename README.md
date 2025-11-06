# obstacle-avoidence-using-real-sense

# RealSense Depth Strip Obstacle Detection

## Overview

This project detects and locates obstacles in a specified horizontal strip of an Intel RealSense depth camera image. It uses 3D projection, ground plane estimation via RANSAC, and outlier clustering to robustly identify obstacles above the ground, even in noisy or uneven environments.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Approach & Methodology](#approach--methodology)
- [Parameters & Tuning](#parameters--tuning)
- [Results & Example Output](#results--example-output)
- [Limitations](#limitations)
- [References](#references)
- [License](#license)

---

## Features

- Reads RealSense depth frames and processes a user-defined image strip.
- Converts 2D pixels with depth to 3D (X, Y, Z).
- Fits ground plane using RANSAC.
- Detects and clusters obstacle points significantly above ground.
- Reports mean (center) position and depth of largest obstacle.
- Visualization using OpenCV.
- Tunable for different robot heights and environments.

---

## Requirements

- Python 3.x
- Intel RealSense SDK (`pyrealsense2`)
- `numpy`
- `opencv-python`
- `scikit-learn`

---

## Setup & Installation

1. **Clone the repository:**

---

## Usage

1. Connect your Intel RealSense camera.
2. Run the main detection script:
3. Adjust detection strip, threshold, or clustering parameters directly in the script as needed.

---

## Approach & Methodology

- **Projection:** Valid strip pixels are transformed into 3D using camera intrinsics.
- **Plane Estimation:** RANSAC fits a plane to identify the ground surface within the strip.
- **Outlier Detection:** Depth points above a threshold distance from the plane are flagged as "obstacles."
- **Clustering:** Obstacle points are clustered (e.g., via DBSCAN) to robustly segment objects, ignoring scattered noise.
- **Output:** The largest cluster's centroid and mean depth are reported and visualized.

### Equations
- **3D Projection:** `X = (u - ppx) / fx * z`; `Y = (v - ppy) / fy * z`; `Z = z`
- **Ground Plane:** `Z = aX + bY + c`

---

## Parameters & Tuning

- **Strip Position & Size:** `strip_y`, `strip_height`, `strip_x_start`, `strip_x_end`
- **RANSAC Residual Threshold:** e.g., `0.02` (meters)
- **Clustering:** `DBSCAN` `eps` in pixels, minimum points

---

## Results & Example Output

Here is an example with an obstacle detected (images and console logs):


Visual output marks strip in green, detected obstacle center with a red dot.

---

## Limitations

- May not distinguish obstacles near ground level if ground is highly uneven.
- False positives possible with very rough terrain or poor depth data.
- Best results require some strip height and consistent ground region in the frame.

---

## References

- [RealSense SDK Documentation](https://dev.intelrealsense.com/docs/)
- [scikit-learn RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)
- Any papers or online tutorials that inspired the approach

---


