import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure
from skimage import io
from skimage.transform import resize

import numpy as np

from skimage.feature import daisy
from skimage import data
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors

from skimage import data
from skimage.viewer import ImageViewer
import cv2


img = io.imread('train/1.png')
img = color.rgb2gray(img)
img=resize(img,(128,128))
#image = color.rgb2gray(data.astronaut())
# image=resize(image, (200, 200))

# fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualise=True)

# fig, (ax2) = plt.subplots(1, 1, figsize=(8, 4), sharex=True, sharey=True)

# # ax1.axis('off')
# # ax1.imshow(image, cmap=plt.cm.gray)
# # ax1.set_title('Input image')
# # ax1.set_adjustable('box-forced')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# #ax1.set_adjustable('box-forced')
# plt.show()





#img = data.camera()
# descs, descs_img = daisy(img, step=180, radius=5, rings=2, histograms=6,
#                          orientations=8, visualize=True)
# img=data.astronaut()
# img = color.rgb2gray(img)



detector_extractor1 = ORB(downscale=1.2, n_keypoints=100)
print(img)
img1 = np.array(img)
img1=img1*255

print(img1)
#print(detector_extractor1.detect(img1))
detector_extractor1.detect_and_extract(img1)
des=detector_extractor1.keypoints
print(des)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(des)
descs_num = des.shape[0] * des.shape[1]
plt.show()
cv2.imshow('nkjn',img)
cv2.waitKey()
