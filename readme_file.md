# 🚗 AI Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Facial%20Mesh-0097A7.svg)](https://google.github.io/mediapipe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered real-time driver drowsiness and fatigue detection system designed to monitor drivers and prevent fatigue-induced accidents. The system leverages Deep Learning (**MobileNetV2**) and Computer Vision (**OpenCV & MediaPipe**) to analyze facial landmarks, track eye state, and trigger real-time alerts.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Driver drowsiness is one of the leading causes of traffic accidents worldwide. This system provides an automated real-time solution that continuously tracks a driver's face via a video feed, detects signs of eye closure or fatigue, and immediately alerts the driver with visual and audio warnings before an accident occurs.

---

## ✨ Key Features

- 🎥 **Real-Time Video Stream**: Fast frame processing directly from webcams or video feeds.
- 🎯 **3D Facial Mesh & Eye Tracking**: Uses **MediaPipe Face Mesh** for precise landmark detection under varying lighting conditions.
- 🧠 **Deep Learning Classification**: Fine-tuned **MobileNetV2** convolutional network to accurately classify eye states (*Open* vs *Closed*).
- 🚨 **Instant Audio Alarm**: Sounds a buzzer/alarm when drowsiness is detected continuously for $N$ consecutive frames.
- ⚡ **High Efficiency**: Lightweight model optimized for real-time edge performance with low latency.
- 📊 **Visual Dashboard**: On-screen status display with live FPS, frame counters, and visual warnings.

---

## 🏗️ System Architecture

```
[ Webcam Feed ] ➡️ [ OpenCV Frame Capture ] ➡️ [ MediaPipe Face Landmark Mesh ]
                                                                │
                                                                ▼
[ Audio Alarm Trigger ] ⬅️ [ Temporal Drowsiness Logic ] ⬅️ [ MobileNetV2 Eye Classifier ]
```

1. **Frame Capture**: Continuously reads input frames from standard cameras.
2. **Face & Landmark Localization**: Extracts 468 facial 3D mesh points using MediaPipe.
3. **Region of Interest (ROI)**: Crops precise eye regions for classification.
4. **Classification**: Passes ROI into MobileNetV2 to predict eye status.
5. **Drowsiness Evaluation**: Tracks consecutive closed-eye frames to prevent false alarms.
6. **Alert System**: Emits audio-visual alarms when thresholds are exceeded.

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.8+ | Primary programming language |
| **Deep Learning** | TensorFlow / Keras | Model training, evaluation & real-time inference |
| **Model Architecture** | MobileNetV2 | Lightweight CNN base for fast computer vision tasks |
| **Computer Vision** | OpenCV | Image acquisition, frame processing, and HUD display |
| **Facial Tracking** | MediaPipe | Real-time face detection and 3D facial landmark mesh |

---

## 📂 Project Structure

```
drowsiness-detection-system/
├── assets/
│   ├── alert.wav             # Audio alert sound file
│   └── preview.gif           # Project demo preview
├── models/
│   └── drowsiness_mobilenetv2.h5  # Pre-trained Keras model weights
├── src/
│   ├── __init__.py
│   ├── detector.py           # Face mesh & eye region extractor
│   ├── alert_system.py       # Audio & visual alert handler
│   └── config.py             # System parameters & threshold settings
├── main.py                   # Main script for real-time detection
├── train.py                  # Script for model training & evaluation
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python `3.8` or higher installed.
- A functional camera/webcam connected to your computer.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/drowsiness-detection-system.git
cd drowsiness-detection-system
```

### 2. Create a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running Real-time Detection
To run the detection system using your default camera:
```bash
python main.py
```
*Press **`q`** to safely terminate the detection process.*

### Model Training (Optional)
To retrain or fine-tune the MobileNetV2 architecture on a custom dataset:
```bash
python train.py --epochs 30 --batch_size 32
```

---

## 🧠 Model Architecture

The classifier is built on top of **MobileNetV2** (pre-trained on ImageNet), fine-tuned specifically for binary eye-state classification (`Open` vs `Closed`).

- **Input Dimensions**: $224 \times 224 \times 3$
- **Optimizer**: Adam ($\text{learning rate} = 10^{-4}$)
- **Loss Function**: Binary Cross-Entropy
- **Model Size**: ~14 MB (suitable for deployment on edge devices like Raspberry Pi)

---

## 🔗 Links & Demo

- **GitHub Repository**: [Add GitHub link]
- **Live Demo**: [Add Demo link]

---

## 🗺️ Roadmap

- [ ] Add head pose estimation (pitch, yaw, roll) to detect driver distraction.
- [ ] Add yawn detection using Mouth Aspect Ratio (MAR).
- [ ] Convert model to TensorFlow Lite (`.tflite`) for deployment on embedded devices.
- [ ] Create a web platform interface using Streamlit or FastAPI.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.