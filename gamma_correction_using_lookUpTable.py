# gamma correction formula = (image/255)^(gamma) * 255
import cv2 as cv
import numpy as np

img = cv.imread('cv2_images/gammacorrection.jpg')

h, w, c = img.shape
img = cv.resize(img, (int(w/2), int(h/2)))

lookUpTable = np.zeros((1,256), np.uint8)

gamma = 0.5
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(np.int32(i) / 255.0, gamma) * 255, 0, 255)

bright_img = cv.LUT(img, lookUpTable)

print(lookUpTable)
cv.imshow('image', bright_img)
cv.waitKey(0)
cv.destroyAllWindows()