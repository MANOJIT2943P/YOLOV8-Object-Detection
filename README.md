<!-- ---
title: YOLOv8 Object Detector with Audio Alerts
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
--- -->

# 📷 YOLOv8 Object Detection with Audio Alerts

Real-time object detection using YOLOv8 ONNX model with text-to-speech alerts.

## Features
- 🎯 4 object classes: Stair, People, Door, Pothole
- 🔊 Audio alerts for detected objects
- 📹 ESP32-CAM and IP camera support
- ⚡ Fast inference with ONNX Runtime

## Usage
1. Enter your camera URL (ESP32-CAM or IP camera)
2. Click "Start Detection"
3. Get real-time detections with audio alerts!

## Supported Camera URLs
- ESP32-CAM: `http://192.168.1.100:81/stream`
- RTSP: `rtsp://username:password@ip:port/stream`
- USB Webcam: `0`