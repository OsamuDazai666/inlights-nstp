# gamma correction formula = (image/255)^(gamma) * 255
import cv2 as cv
import numpy as np

img = cv.imread('cv2_images/gammacorrection.jpg')

h, w, c = img.shape
img = cv.resize(img, (int(w/2), int(h/2)))

gamma = 0.5
bright_img = pow(np.int32(img) / 255.0, gamma) * 255
bright_img = np.clip(bright_img, 0, 255)
bright_img = np.uint8(bright_img)

cv.imshow('bright_img', bright_img)
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
