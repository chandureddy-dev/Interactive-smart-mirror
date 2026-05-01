import cv2

# Load models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Gender labels
GENDER_LIST = ['Male', 'Female']

# Hairstyle suggestions dictionary
HAIRSTYLES = {
    "Male": {
        "Oval": ["Quiff", "Pompadour", "Undercut"],
        "Round": ["High Volume", "Faux Hawk", "Side Part"],
        "Square": ["Buzz Cut", "Crew Cut", "Slick Back"],
        "Heart": ["Fringe", "Comb Over", "Low Fade"]
    },
    "Female": {
        "Oval": ["Long Layers", "Blunt Bob", "Beach Waves"],
        "Round": ["High Ponytail", "Side Bangs", "Straight Long"],
        "Square": ["Wavy Bob", "Layered Lob", "Side Part"],
        "Heart": ["Curtain Bangs", "Soft Waves", "Side-Swept"]
    }
}

def get_face_shape(w, h):
    ratio = w / h
    if abs(w - h) < 20:
        return "Square"
    elif ratio >= 0.95:
        return "Round"
    elif 0.85 <= ratio < 0.95:
        return "Oval"
    else:
        return "Heart"

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Get face ROI and resize
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = x + w + 10
        y2 = y + h + 10
        face_img = frame[y1:y2, x1:x2]

        try:
            face_img = cv2.resize(face_img, (227, 227))
        except:
            continue

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263, 87.7689, 114.896), swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        face_shape = get_face_shape(w, h)
        suggestions = HAIRSTYLES.get(gender, {}).get(face_shape, ["Classic"])

        # Display rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display gender and shape
        text1 = f"Gender: {gender}"
        text2 = f"Face Shape: {face_shape}"
        text3 = f"Hairstyles: {', '.join(suggestions)}"

        # Add text lines
        cv2.putText(frame, text1, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, text2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, text3, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show frame
    cv2.imshow("Smart Mirror Interface", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
