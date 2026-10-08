# Hand-Tracked Voxel Builder

A real-time hand-tracking voxel editor built in Python using OpenCV and MediaPipe. The project lets you create and manipulate a 3D voxel grid using hand gestures captured from a webcam.

## Overview

This application uses a webcam and computer vision to detect hand landmarks and translate gestures into actions such as:
- placing voxels
- changing the active layer
- resizing the brush
- manipulating a cube
- rotating the camera
- visualizing the final 3D voxel structure

The project is designed for interactive 3D modeling using natural hand movements instead of a mouse or keyboard.

## Features

- Live webcam hand tracking
- Gesture-based voxel placement
- Layer selection and editing
- Brush size control via pinch gesture
- Cube manipulation mode
- Camera rotation using two open hands
- Final 3D voxel visualization using Matplotlib

## Technologies Used

- Python
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- Matplotlib
- Matplotlib 3D toolkit

## Requirements

- Python 3.8+
- Webcam
- A working OpenCV + MediaPipe environment
- Required Python packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `matplotlib`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/printhelloearth/Simple-CV-.git
   cd Simple-CV-
