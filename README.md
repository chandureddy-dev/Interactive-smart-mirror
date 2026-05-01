# Interactive-smart-mirror
Computer vision-based smart mirror integrating facial analysis, gesture control, and IoT-based environmental monitoring using Raspberry Pi, OpenCV, and TensorFlow.

## 📌 Project Overview
It the is an intelligent and interactive smart mirror that integrates  
**Artificial Intelligence (AI), Computer Vision, and Internet of Things (IoT)** technologies  
to enhance daily personal care and smart living.

Unlike traditional mirrors, this system provides **real-time environmental monitoring**,  
**face analysis**, **emotion detection**, **gesture-based interaction**, and  
**personalized grooming recommendations**.

---

## 🎯 Key Features
- 👤 **Face Detection & Analysis**
  - Gender estimation
  - Emotion recognition
  - Face-shape identification

- 💇 **AI-Based Hairstyle Recommendation**
  - Suggestions based on gender, face shape, and emotion

- ✋ **Gesture Control**
  - Palm detection for hands-free interaction

- 🌡️ **Environmental Monitoring**
  - Temperature & Humidity (DHT11)
  - Air Quality (MQ-135)
  - Ambient Light (LDR)

- ⚠️ **Alert System**
  - LED and buzzer alerts
  - Automatic fan activation on high temperature

- 💡 **Automatic Brightness & Cooling Control**

- 🖥️ **Real-Time Smart Display**
  - Time, sensor data, alerts, and AI results

---

## 🛠️ Technologies Used

### 🔧 Hardware Components
- Raspberry Pi 4 Model B  
- DHT11 Temperature & Humidity Sensor  
- MQ-135 Gas Sensor  
- LDR Sensor  
- USB Webcam  
- Relay Module  
- Cooling Fan  
- LEDs and Buzzer  
- 7-inch HDMI Display  
- Two-Way Acrylic Mirror  
- 32GB Micro SD Card  

---

### 💻 Software & Tools
- **Programming Language:** Python 3  
- **Operating System:** Raspberry Pi OS  
- **Libraries & Frameworks:**
  - OpenCV
  - TensorFlow / Keras
  - MediaPipe
  - NumPy
  - Adafruit_DHT
  - RPi.GPIO
- **IDE:** Visual Studio Code  

---

## 🧠 System Architecture
The system follows a **layered architecture**:

1. **Input Layer**
   - Sensors (DHT11, MQ-135, LDR)
   - USB Webcam

2. **Processing Layer**
   - Raspberry Pi (AI inference & decision making)

3. **Output Layer**
   - HDMI Display
   - LEDs
   - Buzzer
   - Cooling Fan (Relay controlled)

---

## ⚙️ Working of the System
1. Sensors collect temperature, humidity, gas, and light data  
2. Webcam captures the user’s face  
3. AI models analyze:
   - Gender
   - Emotion
   - Face shape  
4. Personalized hairstyle suggestions are generated  
5. Gesture detection enables hands-free control  
6. Alerts are triggered for unsafe conditions  
7. Results are displayed in real time on the smart mirror  

---

## ▶️ How to Run the Project

### ✅ Prerequisites
- Raspberry Pi with Raspberry Pi OS
- Python 3 installed
- Camera and sensors connected properly

## 📊 Testing & Performance
- **Face detection latency:** < 300 ms
- **Gesture recognition response:** < 1 second
- **Sensor refresh interval:** 2 seconds
- **Continuous runtime tested:** 6+ hours

---

## 🏆 Achievements
- Participated in **CICADA 2025 Project Expo**
- Participated in **InnovateX 2025 Technical Fest**
- Recognized for **innovation and practical implementation**

---

## 🚀 Future Enhancements
- Voice assistant integration (**Google Assistant / Alexa**)
- Cloud & mobile application support
- Biometric authentication (**Fingerprint / Iris**)
- **Augmented Reality (AR)** virtual try-on
- Health & fitness monitoring features
- Smart home automation integration
- Multi-language support




