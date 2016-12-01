import cv2

image = cv2.imread('train/121.png')
resized_image = cv2.resize(image, (256, 256))
cv2.imwrite('121.png',resized_image)
#cv2.imshow('Resized Image',resized_image)
#cv2.waitKey(0)