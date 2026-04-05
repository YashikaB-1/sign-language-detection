# 🤟 Sign Language Detection System

## 📌 Project Overview

The **Sign Language Detection System** is a real-time application that detects hand gestures using a webcam and converts them into meaningful text. This project helps bridge the communication gap between hearing-impaired individuals and others.

---

## 🎯 Objectives

* Detect hand gestures using computer vision
* Convert sign language into text in real-time
* Provide an interactive and user-friendly interface

---

## 🚀 Features

* 📷 Real-time gesture detection using webcam
* 🧠 Machine Learning model for prediction
* ⚡ Fast and responsive Flask web application
* 🎥 Live video streaming with prediction output

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS
* **Backend:** Python (Flask)
* **Libraries:** OpenCV, NumPy, TensorFlow/Keras
* **Model:** LSTM / Deep Learning Model

---

## 📂 Project Structure

```
SignLanguage/
│
├── app.py
├── templates/
│   └── index.html
├── scripts/
│   ├── hand_detection.py
│   ├── realtime_prediction.py
│   ├── train_model.py
│   └── ...
├── models/        (ignored)
├── dataset/       (ignored)
├── .gitignore
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/YashikaB-1/sign-language-detection.git
cd sign-language-detection
```

---

### 2️⃣ Create virtual environment

```
python -m venv asl_env
asl_env\Scripts\activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the project

```
python app.py
```

---

### 5️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

## 📊 Model Information

* Model trained using hand landmark sequences
* Uses deep learning (LSTM) for gesture classification
* Dataset includes multiple sign language gestures

---

## ⚠️ Note

* Dataset and trained model files are not included due to size
* You can retrain the model using provided scripts

---

## 🌟 Future Enhancements

* Add more gesture classes
* Improve accuracy using advanced models
* Add speech output
* Deploy as a web/mobile app

---

## 👩‍💻 Author

**Yashika B**
GitHub: https://github.com/YashikaB-1

---

## 🙌 Acknowledgement

This project is developed for academic and research purposes to assist communication using AI.

---
