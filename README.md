# License Plate Detection using YOLOv3

## Introduction

This project uses **YOLOv3 (You Only Look Once)** for **real-time license plate detection** from images and videos. It also integrates **Optical Character Recognition (OCR)** with **Tesseract** to extract text from detected plates. The system can process video streams, images, or live camera feeds.

---

## Features

- Real-time license plate detection
- YOLOv3-based object detection
- Tesseract OCR for text recognition
- Supports images and videos
- Configurable confidence threshold for detection
- Non-Maximum Suppression (NMS) for improved accuracy

---

## Configuration

### YOLOv3 Configuration Files:

darknet-yolov3.cfg → Model configuration file
classes.names → List of class labels
lapi.weights → Pre-trained YOLOv3 model (Download separately)

### Parameters:

confThreshold → Set confidence threshold for object detection (default: 0.5)
nmsThreshold → Set Non-Maximum Suppression (default: 0.4)
