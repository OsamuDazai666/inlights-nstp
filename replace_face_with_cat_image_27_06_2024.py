import cv2

def replace_face_with_cat(frame, replacement_img, threshold, expand=30):
    faces = classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        replacement_img = cv2.resize(replacement_img, (w, h))

        roi = frame[y:y+h, x:x+w] # region of interest is the face

        replace_img2gray = cv2.cvtColor(replacement_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(replace_img2gray, threshold, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img_fg = cv2.bitwise_and(replacement_img, replacement_img, mask=mask_inv)
        combined = cv2.add(img_bg, img_fg)

        frame[y:y+h, x:x+w] = combined

    return frame
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('cv2_models/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    replacement_img = cv2.imread('cv2_images/cat2.jpg')
    faces = classifier.detectMultiScale(frame)

    frame = replace_face_with_cat(frame, replacement_img, 200, expand=30)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
