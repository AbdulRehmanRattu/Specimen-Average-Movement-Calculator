# Automated Microscopic Specimen Motion Detection and Motility Analysis Platform

## Overview

Microscopic specimen motility tracking and velocity profiling are essential procedures in reproductive biology, cellular immunology, and microbiological assays. Manual microscopic counting is labor-intensive, operator-dependent, and prone to significant observational error.

This project develops an automated desktop computer vision application for real-time microscopic specimen detection, motion contour tracking, and kinetic velocity analysis in biological video streams. Built using Python, OpenCV, and Tkinter with a multi-threaded execution architecture, the software processes multi-frame video inputs to segment active specimens, compute motility counts, and display live tracking overlays without UI freezing.

---


---

## Problem Statement

Microscopic biological specimen counting and motility tracking in medical diagnostics and microbiology assays are traditionally performed through manual microscope observation. This manual approach is labor-intensive, operator-dependent, and prone to substantial counting variance. Biological research laboratories require an automated desktop computer vision application capable of processing real-time microscope video streams, isolating active specimen motility contours, and computing velocity distributions without computational lag or UI freezing.

## System Architecture and Workflow

```
[ Microscopic Video Feed (.mp4 / .avi) ]
 |
 v
[ Multi-Threaded Video Decoding & Frame Buffer Ingestion ]
 |
 v
[ Computer Vision Motion Detection Pipeline ]
 + Grayscale Conversion & Gaussian Blur Noise Filtering
 + Temporal Frame Differencing & Background Subtraction
 + Adaptive Thresholding & Morphological Dilation
 + Contour Extraction & Bounding Box Localization
 |
 v
[ Quantitative Kinematic Metric Computation ]
 + Motility Density & Particle Velocity Estimation
 + Trajectory Tracking & Spatial Coordinate Logging
 |
 v
[ Interactive Tkinter Graphical Dashboard & Live Video Canvas ]
```

---

## Key Features

- **Real-Time Motion Segmentation**: Employs temporal frame differencing and background subtraction to detect active microscopic specimen motility against stationary particulate debris.
- **Morphological Contour Filtering**: Isolates individual biological cells using adaptive thresholding, morphological closing/dilation, and minimum area thresholds.
- **Multi-Threaded UI Architecture**: Separates video frame computation into asynchronous background threads (`threading.Thread`), maintaining a responsive Tkinter GUI during high-resolution video processing.
- **Live Video Canvas Overlay**: Renders bounding box trajectories and centroid markers directly onto the live canvas interface.
- **Standalone Windows Executable**: Bundled with a pre-configured PyInstaller specification (`app.spec`) to generate a portable Windows binary (`dist/app.exe`).

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Programming Language** | Python 3.8+ |
| **Computer Vision Engine** | OpenCV (`cv2`) |
| **GUI Framework** | Tkinter, TTK Themed Widgets |
| **Threading Model** | Python `threading` Asynchronous Video Loop |
| **Image Formatting** | Pillow (`PIL.Image`, `PIL.ImageTk`) |
| **Executable Packaging** | PyInstaller (`app.spec` -> `app.exe`) |

---

## Project Structure

```
specimen-motion-detection-app/
├── Specimen Desktop App/
│ ├── app.py # Multi-threaded Tkinter application & CV tracking engine
│ ├── project.py # Script-based processing and analysis utilities
│ ├── app.spec # PyInstaller standalone build configuration
│ ├── sperm.mp4 # Microscopic motility sample video
│ ├── simple.mp4 # Benchmark single-cell trajectory video
│ ├── looped.mp4 # High-density cyclic motility sample video
│ └── dist/
│ └── app.exe # Compiled portable Windows binary
├── requirements.txt # Python runtime dependencies
└── README.md # System documentation
```

---

## Installation and Environment Setup

### 1. Clone Repository
```bash
git clone https://github.com/AbdulRehmanRattu/Specimen-Motion-Detection-App.git
cd Specimen-Motion-Detection-App
```

### 2. Configure Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Requirements Specification (`requirements.txt`)
```
opencv-python>=4.8.0
numpy>=1.23.0
pillow>=9.5.0
pyinstaller>=5.13.0
```

---

## Usage Guide

### 1. Launch GUI Application
```bash
cd "Specimen Desktop App"
python app.py
```

### 2. Operational Workflow
1. Click **Browse** to select a specimen video file (`sperm.mp4`, `simple.mp4`, or any standard `.mp4`/`.avi` microscope recording).
2. Click **Run** to initiate automated background motion tracking.
3. Observe live specimen contours on the video canvas and monitor real-time kinematic metrics in the logging console.
