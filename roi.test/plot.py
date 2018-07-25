import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

# img1 = './liang.jpg'
img1 = './testdata/000032.jpg'
img22 = './stn_later0.jpg'
img33 = './stn_later1.jpg'
img44 = './stn_later2.jpg'
img55 = './stn_later3.jpg'
img2 = './roi_pool_later0.jpg'
img3 = './roi_pool_later1.jpg'
img4 = './roi_pool_later2.jpg'
img5 = './roi_pool_later3.jpg'

img1_data = cv.imread(img1)
img2_data = cv.imread(img2)
img3_data = cv.imread(img3)
img4_data = cv.imread(img4)
img5_data = cv.imread(img5)
img22_data = cv.imread(img22)
img33_data = cv.imread(img33)
img44_data = cv.imread(img44)
img55_data = cv.imread(img55)

gt_boxes = np.array([[104, 78, 375, 183],\
                    [133, 88, 197, 123],\
                    [195, 180, 213, 229],\
                    [26, 189, 44, 238]])
gt_boxes = gt_boxes[np.newaxis, :]

plt.subplot(521)
plt.imshow(img1_data)
plt.axis('off')
# box = gt_boxes[0][0]
for box in gt_boxes[0]:
    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],\
                                    fill=False, edgecolor='r', linewidth=3))
# plt.show()

# plt.subplot(321)
# plt.imshow(img1_data)
plt.subplot(523)
plt.imshow(img2_data)
plt.axis('off')
plt.subplot(524)
plt.imshow(img3_data)
plt.axis('off')
plt.subplot(525)
plt.imshow(img4_data)
plt.axis('off')
plt.subplot(526)
plt.imshow(img5_data)
plt.axis('off')
plt.subplot(527)
plt.imshow(img22_data)
plt.axis('off')
plt.subplot(528)
plt.imshow(img33_data)
plt.axis('off')
plt.subplot(529)
plt.imshow(img44_data)
plt.axis('off')
plt.subplot(5,2,10)
plt.imshow(img55_data)
plt.axis('off')
plt.show()
