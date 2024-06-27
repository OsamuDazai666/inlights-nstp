import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('cv2_models/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    faces = classifier.detectMultiScale(frame)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
