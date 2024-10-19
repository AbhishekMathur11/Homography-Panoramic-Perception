# Homography-Panoramic-Perception

## Overview

This project implements augmented reality using planar homographies. It includes feature detection, matching, and homography computation to overlay virtual objects onto real-world images.

## Key Components

### Feature Detection and Matching

- Implements FAST corner detection
- Uses BRIEF descriptors for feature matching
- Includes functions for:
  - Corner detection
  - Descriptor computation
  - Feature matching

### Homography Computation

- Implements Direct Linear Transform (DLT) for homography estimation
- Includes RANSAC for robust estimation

### AR Overlay

- Applies computed homography to overlay virtual objects on input images

## Results

![Feature Matching Result](path/to/feature_matching_result.png)

*Figure 1: Feature matching between two images*

![AR Overlay Result](path/to/ar_overlay_result.png)

*Figure 2: Virtual object overlaid on input image*

