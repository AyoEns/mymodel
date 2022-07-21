import copy
import math
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import xml.etree.ElementTree as ET
from centernet import CenterNet
from utils.utils import get_classes
from scipy.optimize import linear_sum_assignment
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":
    mode = "predict"
    heatmap_save_path = "model_data/heatmap_vision.png"

    centernet = CenterNet(heatmap=True if mode == "heatmap" else False)

    crop = False
    count = False
    classes_path = "./model_data/myclasses.txt"
    classes, _ = get_classes(classes_path)

    if mode == "predict":
        while True:
            imgpath = input('Input image filename:')
            try:
                image = Image.open(imgpath)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, result = centernet.detect_image(image, crop=crop, count=count)
                # r_image.show()
                # print(imgpath)
                image1 = cv2.imread(imgpath)
                zero_map = copy.deepcopy(image1)
                # print(result)
                result_point = []
                for i in range(len(result)):
                    x = (result[i][0] + result[i][2]) // 2
                    y = (result[i][1] + result[i][3]) // 2
                    x = int(x)
                    y = int(y)
                    result_point.append([y,x])
                    print(x, y)
                    cv2.circle(image1, (y, x), radius=2, color=(0, 0, 0), thickness=-1)

                #
                cv2.imwrite("1.jpg", image1, [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imshow("1", image1)
                cv2.waitKey()
                print("Done!")
                # xmlpath = imgpath.replace("JPEGImage", "Annotations")
                # xmlpath = xmlpath.replace(".jpg", ".xml")
                #
                # in_file = open(xmlpath, encoding='utf-8')
                # tree = ET.parse(xmlpath)
                # root = tree.getroot()
                # GT_point = []
                # for obj in root.iter('object'):
                #     difficult = 0
                #     if obj.find('difficult') != None:
                #         difficult = obj.find('difficult').text
                #     cls = obj.find('name').text
                #     if cls not in classes or int(difficult) == 1:
                #         continue
                #     cls_id = classes.index(cls)
                #     # xmlbox = obj.find('bndbox')
                #     # b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
                #     xml_points = obj.find('Keypoints_XY')
                #     x = int(xml_points.find("keypoints_x").text)
                #     y = int(xml_points.find("keypoints_y").text)
                #     cv2.circle(image1, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
                #     GT_point.append([x,y])


                # T = []
                # for i in range(len(GT_point)):
                #     z = []
                #     for j in range(len(result_point)):
                #         X1 = GT_point[i][0] - result_point[j][0]
                #         Y1 = GT_point[i][1] - result_point[j][1]
                #         # X = X1 / 20000
                #         # Y = Y1 / 20000
                #         X = pow(X1, 2)
                #         Y = pow(Y1, 2)
                #         cost_ = math.sqrt(X + Y)
                #         z.append(np.asarray(cost_))
                #     T.append(z)
                #
                # cost_mat = np.array(T)
                # work_idx_ls, pokeman_idx_ls = linear_sum_assignment(cost_mat)
                # match_result = list(zip(work_idx_ls, pokeman_idx_ls))
                # print(match_result)
                # print("GT points numbers:", len(GT_point))
                # print("Predict points numbers:", len(result_point))
                # print("match result numbers:", len(match_result))
                # text_points_nums = 250
                # # cv2.circle(zero_map, (GT_point[work_idx_ls[text_points_nums]][0], GT_point[work_idx_ls[text_points_nums]][1]), radius=5, color=(255,255,255), thickness=-1)
                # # cv2.circle(zero_map, (result_point[pokeman_idx_ls[text_points_nums]][0], result_point[pokeman_idx_ls[text_points_nums]][1]), radius=5, color=(0, 0, 0), thickness=-1)
                # colors1 = (0,0,0)
                # colors2 = (255, 255, 255)
                # save_path = "./match_points_image"
                # sum = 0
                # count = 1
                # TP = 0
                # FP = 0
                #
                # for i in range(len(match_result)):
                #     zero_map_temp = copy.deepcopy(zero_map)
                #     cv2.circle(zero_map_temp, (GT_point[work_idx_ls[i]][0], GT_point[work_idx_ls[i]][1]), radius=4, color=colors1, thickness=1)
                #     cv2.circle(zero_map_temp, (result_point[pokeman_idx_ls[i]][0], result_point[pokeman_idx_ls[i]][1]), radius=4, color=colors2, thickness=1)
                #     names = os.path.join(save_path, f"{i}.jpg")
                #     cv2.imwrite(names, zero_map_temp, [cv2.IMWRITE_JPEG_QUALITY, 100])
                #     x_distance = GT_point[work_idx_ls[i]][0] - result_point[pokeman_idx_ls[i]][0]
                #     y_distance = GT_point[work_idx_ls[i]][1] - result_point[pokeman_idx_ls[i]][1]
                #     oula = math.sqrt(x_distance ** 2 + y_distance ** 2)
                #     print(oula)
                #     if oula < 15:
                #         TP += 1
                #     else:
                #         FP += 1
                # P = float(TP / (TP + FP))
                # Recall = float(TP / len(GT_point))
                # F1 = 2 * (P * Recall) / (P + Recall)
                # print(f"P:{P * 100.0}%, Recall:{Recall * 100.0}%, F1:{F1 * 100}%")
                # # print("pixel error", float(sum / count))
                # # cv2.imshow("show_points_match", zero_map)
                # # cv2.waitKey()
                # print("Done!")