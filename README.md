# Fall Detection System

An automated fall detection system using computer vision, built with Roboflow's pre-trained model and deployed on Streamlit.

## Project Overview

This project implements an automated fall detection system that can analyze video footage or live camera feeds to identify when a person falls. The system uses a pre-trained object detection model to classify human poses as either "standing" or "falling" with confidence scores.

**Live Demo:** [https://fall-detection-roboflow-computer-vision.streamlit.app/](https://fall-detection-roboflow-computer-vision.streamlit.app/)

## Details

### Model Information
- **Model Type:** Roboflow 3.0 Object Detection (Fast)
- **Dataset:** fall-detection-mbldh with 7,178 training images
- **Checkpoint:** COCO pre-trained weights
- **Confidence Threshold:** Adjustable (default: 0.2)

### Stack
- **Computer Vision:** OpenCV (cv2)
- **API Integration:** Roboflow REST API
- **Web Framework:** Streamlit
- **Deployment:** Streamlit Cloud

## Features

- **Video Upload Analysis:** Process MP4, AVI, MOV, and MKV video files
- **Real-time Webcam Detection:** Capture and analyze images from webcam
- **Interactive Interface:** Adjustable confidence threshold slider
- **Visual Annotations:** Bounding boxes with color-coded labels
- **Download Results:** Export annotated videos
- **Progress Tracking:** Real-time processing status

### Detection Legend
- 🟢 **Green boxes:** Person standing
- 🔴 **Red boxes:** Person falling
- 🟡 **Yellow boxes:** Unknown/other classes
- **Confidence scores:** Displayed as percentages (0-100%)

## 📱 How to Use the Web Application

### Option 1: Video Upload
1. Visit the [live application](https://fall-detection-roboflow-computer-vision.streamlit.app/)
2. Select "Upload Video" option
3. Adjust confidence threshold if needed (sidebar)
4. Upload your video file (supported: MP4, MOV, AVI, MKV)
5. Click "Analyze Video"
6. Wait for processing to complete
7. View the annotated result and download if needed

### Option 2: Webcam Analysis
1. Select "Webcam" option
2. Adjust confidence threshold if needed
3. Click "Take a picture"
4. Allow camera access when prompted
5. Take a photo to analyze
6. View the instant analysis results

## 💻 Local Installation & Setup

### Prerequisites
- Webcam (optional, for real-time detection)

### WATCH DEMO VIDEO HERE
[Watch Demo Video](assets/fall_output_annotatedv1.mp4)


## License

This project is licensed under the MIT License

## 🙏 Acknowledgments

- **Roboflow** for providing the pre-trained fall detection model
- **Streamlit** for web framework
- **OpenCV** community for computer vision tools


**⭐ If you found this project useful, please consider giving it a star!**