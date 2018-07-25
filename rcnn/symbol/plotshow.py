import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

# img1 = './liang.jpg'
img1 = './resized.jpg'
img2 = './stn_later.jpg'

img1_data = cv.imread(img1)
img2_data = cv.imread(img2)

plt.subplot(211)
plt.imshow(img1_data)
plt.subplot(212)
plt.imshow(img2_data)
plt.show()