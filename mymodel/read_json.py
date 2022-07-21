import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
load_f = open("dataset/before/Image0292 12-23-07.json", "r")
load_dict = json.load(load_f)
line_points = []
for i in load_dict["shapes"]:
    label_name = i["label"]
    points = i["points"]
    line_points.append([label_name, points])

w = load_dict["imageWidth"]
h = load_dict["imageHeight"]
h_line = h // 5
h_ = np.arange(50, h, h_line)
print(h_)
img = cv2.imread("dataset/before/Image0292 12-23-07.jpg")
keypoints = []
for i in range(len(line_points)):
    x = []
    y = []
    x1 = int(line_points[i][1][0][0])
    y1 = int(line_points[i][1][0][1])
    x2 = int(line_points[i][1][1][0])
    y2 = int(line_points[i][1][1][1])
    cv2.line(img, (x1, y1), (x2, y2), color=(0,0,0), thickness=3)
    x.append(x1)
    x.append(x2)
    y.append(y1)
    y.append(y2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_ = (h_ - z[1]) / z[0]
    for j in range(len(x_)):
        cv2.circle(img, (int(x_[j]), int(h_[j])), radius=5, color=(255,255,255), thickness=-1)
        keypoints.append([int(x_[j]), int(h_[j])])

cv2.imshow("img", img)
print(len(keypoints))
cv2.waitKey()

