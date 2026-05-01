import cv2
from datetime import datetime

# Load face detection and gender model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']

# Start working webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("✅ Webcam opened. Starting Smart Mirror...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read frame.")
        continue

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("😐 No face detected.")

    for (x, y, w, h) in faces:
        x1, y1 = max(0, x - 10), max(0, y - 10)
        x2, y2 = min(frame.shape[1], x + w + 10), min(frame.shape[0], y + h + 10)
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            continue

        try:
            resized = cv2.resize(face_img, (227, 227))
        except:
            continue

        blob = cv2.dnn.blobFromImage(resized, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = GENDER_LIST[preds[0].argmax()]

        # Terminal Print
        print("🧑 Detected Gender:", gender)

        # Frame Label
        label = f"🧑 Gender: {gender}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Smart Mirror UI
    now = datetime.now()
    date_str = now.strftime("📅 %d %b %Y")
    time_str = now.strftime("⏰ %I:%M:%S %p")
    temp_str = "🌡️ Temp: 25°C"
    humid_str = "💧 Humidity: 65%"

    # Overlay Info
    cv2.putText(frame, date_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
    cv2.putText(frame, time_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
    cv2.putText(frame, temp_str, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
    cv2.putText(frame, humid_str, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 200), 2)

    # Show Frame
    cv2.imshow("🪞 Smart Mirror - Gender Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting Smart Mirror...")
        break

cap.release()
cv2.destroyAllWindows()
