import cv2#camera related
import numpy as np
import Adafruit_DHT#dht11 library
import RPi.GPIO as GPIO#raspberry pi  pins related library
import spidev#a nalog to digital conversion reading related library
import time
import threading
from datetime import datetime
from keras.models import load_model
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load models
emotion_model = load_model('/home/pi/Desktop/folder/emotion_model.h5', compile=False)
gender_net = cv2.dnn.readNetFromCaffe(
    '/home/pi/Desktop/folder/gender_deploy.prototxt',
    '/home/pi/Desktop/folder/gender_net.caffemodel'
)
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/folder/haarcascade_frontalface_default.xml')

# Labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
GENDERS = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Sensor & GPIO setup
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4
RELAY_PIN, LED_PIN, BUZZER_PIN = 17, 26, 22
TEMP_THRESHOLD, LDR_THRESHOLD, GAS_THRESHOLD = 33, 600, 500

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

# Global sensor data (thread updated)
sensor_data = {"temp": None, "hum": None, "gas": 0, "ldr": 0}

# Helper functions
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

def sensor_thread():
    while True:
        humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)
        gas_value = read_adc(0)
        ldr_value = read_adc(1)
        sensor_data["temp"] = temperature
        sensor_data["hum"] = humidity
        sensor_data["gas"] = gas_value
        sensor_data["ldr"] = ldr_value

        # Relay control
        if temperature is not None and temperature > TEMP_THRESHOLD:
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        # LED & buzzer control
        GPIO.output(LED_PIN, GPIO.HIGH if ldr_value > LDR_THRESHOLD else GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.HIGH if gas_value > GAS_THRESHOLD else GPIO.LOW)

        time.sleep(2)  # update every 2 sec

def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    reshaped = resized.reshape(1, 64, 64, 1) / 255.0
    result = emotion_model.predict(reshaped, verbose=0)
    return EMOTIONS[np.argmax(result)]

def get_face_shape(w, h):
    ratio = w / float(h)
    if ratio > 1.2:
        return "Round"
    elif 0.9 < ratio <= 1.2:
        return "Oval"
    else:
        return "Square"

def get_hairstyle(gender, face_shape, emotion):
    male = {
        "Round": ["Faux Hawk", "Spiky Fade", "Side Part"],
        "Oval": ["Crew Cut", "Comb Over", "Undercut"],
        "Square": ["Buzz Cut", "Pompadour", "Slick Back"]
    }
    female = {
        "Round": ["Long Layers", "Side Bangs", "Textured Bob"],
        "Oval": ["Wavy Lob", "Straight Mid-length", "Loose Curls"],
        "Square": ["Pixie Cut", "Top Knot", "Braided Crown"]
    }
    styles = male if gender == "Male" else female
    options = styles.get(face_shape, ["Classic Look"])
    return options[EMOTIONS.index(emotion) % len(options)]

def get_emotion_tip(emotion):
    return {
        "Angry": "Breathe and relax your mind.",
        "Disgust": "Step away and clear your space.",
        "Fear": "You’re safe. You're strong.",
        "Happy": "Spread your joy around!",
        "Sad": "You matter. Rest and recharge.",
        "Surprise": "Unexpected can be wonderful!",
        "Neutral": "Balance is powerful. Keep going."
    }.get(emotion, "")

# Start sensor thread
threading.Thread(target=sensor_thread, daemon=True).start()

# Camera setup
cap = cv2.VideoCapture(0)

frame_count = 0
last_emotion, last_gender, last_hairstyle, last_tip = "", "", "", ""

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Only run model every 10 frames for speed
        if frame_count % 10 == 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                last_emotion = detect_emotion(face_img)
                resized_face = cv2.resize(face_img, (227, 227))
                blob = cv2.dnn.blobFromImage(resized_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                last_gender = GENDERS[gender_net.forward()[0].argmax()]
                face_shape = get_face_shape(w, h)
                last_hairstyle = get_hairstyle(last_gender, face_shape, last_emotion)
                last_tip = get_emotion_tip(last_emotion)
        frame_count += 1

        # Draw sensor data
        t, hmd, gas, ldr = sensor_data["temp"], sensor_data["hum"], sensor_data["gas"], sensor_data["ldr"]
        if t is not None and hmd is not None:
            cv2.putText(frame, f"Temp: {t:.1f} C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Humidity: {hmd:.1f} %", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Sensor Read Fail", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Air: {'Poor' if gas > GAS_THRESHOLD else 'Good'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 180), 2)
        cv2.putText(frame, f"Light: {'LOW' if ldr > LDR_THRESHOLD else 'HIGH'}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

        # Date and time
        now = datetime.now()
        cv2.putText(frame, now.strftime("%d-%b-%Y"), (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)
        cv2.putText(frame, now.strftime("%H:%M:%S"), (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)

        # Draw face info
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{last_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"{last_gender} | {last_hairstyle}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(frame, f"Tip: {last_tip}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 180), 1)

        cv2.imshow("Smart Mirror", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
