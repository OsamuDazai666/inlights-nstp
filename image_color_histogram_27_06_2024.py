import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cv2_images/beach.jpg", -1)
h, w, c = img.shape
img = cv2.resize(img, (int(w/2), int(h/2)))
h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
h_array = h.flatten()

fig = plt.figure(figsize=(1, 2))

fig.add_subplot(1, 2, 1)
plt.hist(img.ravel(), 256, [0, 256])
plt.title("rgb")

fig.add_subplot(1, 2, 2)
plt.hist(h_array, bins=180, color='r')
plt.title("hsv")

# fig.tight_layout(pad=5.0)

# set the spacing between subplots
fig.tight_layout()
plt.show()



cv2.imshow("beach", img)
cv2.waitKey(0)
cv2.destroyAllWindows()