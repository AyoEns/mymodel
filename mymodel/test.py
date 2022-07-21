import cv2
from PIL import Image

image = Image.open("VOCdevkit/VOC2007/JPEGImage/mls_result_roi_37_25.jpg")
image.show()
image = cv2.imread("VOCdevkit/VOC2007/JPEGImage/mls_result_roi_37_25.jpg")
cv2.imshow("1", image)
cv2.waitKey()
