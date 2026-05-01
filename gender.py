import cv2

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load gender detection model
gender_proto_path = 'gender_deploy.prototxt'
gender_model_path = 'gender_net.caffemodel'
gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)

# Gender Labels
GENDER_LIST = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Improved cropping with padding and resizing
        x1 = max(0, x - 10)
        y1 = max(0, y - 10)
        x2 = x + w + 10
        y2 = y + h + 10
        face_img = frame[y1:y2, x1:x2]

        try:
            face_img = cv2.resize(face_img, (227, 227))
        except:
            continue  # Skip if resize fails

        # Blob creation
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     (78.4263377603, 87.7689143744, 114.895847746), 
                                     swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Display label and bounding box
        label = f"Gender: {gender}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show video window
    cv2.imshow("Smart Mirror - Gender Detection", frame)
    
    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
