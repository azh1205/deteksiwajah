 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
 [![Made With Python](https://img.shields.io/badge/Made%20With-Python-blue.svg)]()

# 🔍 DeteksiWajah – Face Detection & Digital Forensics Suite

A desktop application built with Python for face verification, live face recognition, digital image forensics, and face database management — powered by [DeepFace](https://github.com/serengil/deepface) and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).

---

## ✨ Features

### 🧑‍🤝‍🧑 Face Verification
- Compare two face images using multiple AI models
- Displays distance score, similarity percentage, and verification result
- Supports threshold customization (e.g., ArcFace)

### 🕵️ Digital Forensics
- **Hash Analysis** – MD5, SHA-1, SHA-256 file hashes
- **EXIF Metadata** – Extract camera, device, and image metadata
- **ELA (Error Level Analysis)** – Detect potential image manipulation
- **GPS Info** – Extract location data from image EXIF
- **Thumbnail Detection** – Check for embedded EXIF thumbnails
- **String Extraction** – Find hidden text strings in binary image files
- **Full Summary Report** – Run all forensic checks at once

### 📷 Live Face Recognition
- Real-time webcam face recognition against a local face database
- Multi-camera support with automatic detection
- Attendance tracking with first-seen timestamps
- Detection history timeline chart
- Export detection history to CSV

### 🗃️ Add to Database
- Capture photos via webcam and save them to the face database
- Automatic face detection before saving
- Organizes photos by person name into subfolders

---

## 🖥️ Requirements

- Python 3.8+
- Webcam (for live recognition and database features)

### Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Key packages used:

| Package | Purpose |
|---|---|
| `deepface` | Face recognition and verification |
| `customtkinter` | Modern dark-themed GUI |
| `opencv-python` | Camera capture and image processing |
| `Pillow` | Image display and manipulation |
| `matplotlib` | Detection history charts |
| `numpy` | Numerical operations |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/azh1205/deteksiwajah.git
cd deteksiwajah
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up the face database

Place face images inside the `face_database/` folder, organized by person name:

```
face_database/
├── Alice/
│   ├── alice_01.jpg
│   └── alice_02.jpg
└── Bob/
    └── bob_01.jpg
```

> You can also use the **"Add to Database"** tab in the app to add faces via webcam.

### 4. Run the application

```bash
python face_gui.py
```

---

## 📁 Project Structure

```
deteksiwajah/
├── face_gui.py          # Main application file
├── live_analysis.py     # Live recognition logic
├── requirements.txt     # Python dependencies
├── face_database/       # Face image database (organized by person)
│   └── facedatahere.txt # Placeholder — add face images here
└── README.md
```

---

## 🤖 Supported AI Models

The following face recognition models are supported for verification and live recognition:

- **VGG-Face** *(default)*
- **Facenet**
- **ArcFace**
- **OpenFace**
- **DeepID**
- **SFace**
- **Dlib** *(optional, requires extra install)*

---

## ⚠️ Notes

- The `face_database/` folder is excluded from version control. Add your own face images locally.
- First launch may take a moment as AI models are downloaded and warmed up automatically.
- For best live recognition performance, use good lighting and a front-facing camera.
- ArcFace uses a custom threshold of `0.50` for stricter matching.

---

## 📄 License

This project is for educational and research purposes.
