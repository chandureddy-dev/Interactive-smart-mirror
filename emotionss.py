import cv2
import numpy as np
from keras.models import load_model

# Load Emotion Detection Model
emotion_model = load_model('/home/pi/Desktop/folder/emotion_model.h5', compile=False)

# Load Gender Detection Model
gender_net = cv2.dnn.readNetFromCaffe(
    '/home/pi/Desktop/folder/gender_deploy.prototxt',
    '/home/pi/Desktop/folder/gender_net.caffemodel'
)

# Constants
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
GENDERS = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier('/home/pi/Desktop/folder/haarcascade_frontalface_default.xml')

# Determine face shape
def get_face_shape(w, h):
    ratio = w / float(h)
    if ratio > 1.2:
        return "Round"
    elif 0.9 < ratio <= 1.2:
        return "Oval"
    else:
        return "Square"

# Hairstyle suggestions
def get_hairstyle(gender, face_shape, emotion):
    male_styles = {
        "Round": ["Faux Hawk", "Spiky Fade", "Side Part"],
        "Oval": ["Crew Cut", "Comb Over", "Undercut"],
        "Square": ["Buzz Cut", "Pompadour", "Slick Back"]
    }
    female_styles = {
        "Round": ["Long Layers", "Side Bangs", "Textured Bob"],
        "Oval": ["Wavy Lob", "Straight Mid-length", "Loose Curls"],
        "Square": ["Pixie Cut", "Top Knot", "Braided Crown"]
    }
    styles = male_styles if gender == "Male" else female_styles
    options = styles.get(face_shape, ["Classic Look"])
    return options[EMOTIONS.index(emotion) % len(options)]

# Emotion tips
def get_emotion_tip(emotion):
    tips = {
        "Angry": "Breathe and relax your mind.",
        "Disgust": "Step away and clear your space.",
        "Fear": "You’re safe. You're strong.",
        "Happy": "Spread your joy around!",
        "Sad": "You matter. Rest and recharge.",
        "Surprise": "Unexpected can be wonderful!",
        "Neutral": "Balance is powerful. Keep going."
    }
    return tips.get(emotion, "")

# Predict emotion
def detect_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    reshaped = resized.reshape(1, 64, 64, 1) / 255.0
    result = emotion_model.predict(reshaped, verbose=0)
    return EMOTIONS[np.argmax(result)]

# Camera Start
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x-10), max(0, y-10)
        x2, y2 = x+w+10, y+h+10
        face_img = frame[y1:y2, x1:x2]

        # Emotion Detection
        emotion = detect_emotion(face_img)

        # Gender Detection
        resized_face = cv2.resize(face_img, (227, 227))
        blob = cv2.dnn.blobFromImage(resized_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender = GENDERS[gender_net.forward()[0].argmax()]

        # Face Shape
        face_shape = get_face_shape(w, h)

        # Hairstyle and Tip
        hairstyle = get_hairstyle(gender, face_shape, emotion)
        tip = get_emotion_tip(emotion)

        # Terminal Output (IDLE or shell)
        print(f"\nDetected:")
        print(f"Gender     : {gender}")
        print(f"Emotion    : {emotion}")
        print(f"Face Shape : {face_shape}")
        print(f"Hairstyle  : {hairstyle}")
        print(f"Tip        : {tip}")

        # Overlay on Frame
        label = f"{gender}, {emotion}, {face_shape}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Style: {hairstyle}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
        cv2.putText(frame, f"Tip: {tip}", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

    cv2.imshow("Smart Mirror Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
