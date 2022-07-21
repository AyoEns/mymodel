import os
import json
from xml.dom.minidom import Document
import cv2
import numpy as np


def CreateXML(save_xml_path, keypoints, jpg_name, jpg_shape):
    doc = Document()
    orderpack = doc.createElement("annotation")
    doc.appendChild(orderpack)
    folder = doc.createElement("folder")
    folder.appendChild(doc.createTextNode("VOC2007"))
    orderpack.appendChild(folder)

    filename_xml = doc.createElement("filename")
    filename_xml.appendChild(doc.createTextNode(f"{jpg_name}"))
    orderpack.appendChild(filename_xml)

    source = doc.createElement("source")

    database = doc.createElement("database")
    database.appendChild(doc.createTextNode("The VOC2007 Database"))
    source.appendChild(database)

    annotation = doc.createElement("annotation")
    annotation.appendChild(doc.createTextNode(f"PASCAL VOC2007"))
    source.appendChild(annotation)

    image = doc.createElement("image")
    image.appendChild(doc.createTextNode(f"flickr"))
    source.appendChild(image)

    flickrid = doc.createElement("flickrid")
    flickrid.appendChild(doc.createTextNode(f" "))
    source.appendChild(flickrid)
    orderpack.appendChild(source)

    owner = doc.createElement("owner")
    flickrid_ = doc.createElement("flickrid")
    flickrid_.appendChild(doc.createTextNode(f" "))
    owner.appendChild(flickrid_)

    name = doc.createElement("name")
    name.appendChild(doc.createTextNode(f" "))
    owner.appendChild(name)
    orderpack.appendChild(owner)

    size = doc.createElement("size")
    width = doc.createElement("width")
    width.appendChild(doc.createTextNode(f"{jpg_shape[1]}"))
    size.appendChild(width)

    height = doc.createElement("height")
    height.appendChild(doc.createTextNode(f"{jpg_shape[0]}"))
    size.appendChild(height)

    depth = doc.createElement("depth")
    depth.appendChild(doc.createTextNode(f"{jpg_shape[2]}"))
    size.appendChild(depth)
    orderpack.appendChild(size)

    segmented = doc.createElement("segmented")
    segmented.appendChild(doc.createTextNode(f"0"))
    orderpack.appendChild(segmented)

    for i in range(len(keypoints)):
        x_ = keypoints[i][0]
        y_ = keypoints[i][1]
        line_index_ = keypoints[i][2]
        in_line_index = keypoints[i][3]
        xmin_ = keypoints[i][0] - 5
        xmax_ = keypoints[i][0] + 5
        ymin_ = keypoints[i][1] - 5
        ymax_ = keypoints[i][1] + 5
        object = doc.createElement("object")

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(f"keypoint"))
        object.appendChild(name)

        pose = doc.createElement("pose")
        pose.appendChild(doc.createTextNode(f"Left"))
        object.appendChild(pose)

        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode(f"0"))
        object.appendChild(truncated)

        difficult = doc.createElement("difficult")
        difficult.appendChild(doc.createTextNode(f"0"))
        object.appendChild(difficult)

        Keypoints_XY = doc.createElement("Keypoints_XY")

        keypoints_x = doc.createElement("keypoints_x")
        keypoints_x.appendChild(doc.createTextNode(f"{str(x_)}"))
        Keypoints_XY.appendChild(keypoints_x)

        keypoints_y = doc.createElement("keypoints_y")
        keypoints_y.appendChild(doc.createTextNode(f"{str(y_)}"))
        Keypoints_XY.appendChild(keypoints_y)
        object.appendChild(Keypoints_XY)

        line_index_xml = doc.createElement("line_index")
        line_index_xml.appendChild(doc.createTextNode(f"{str(line_index_)}"))
        object.appendChild(line_index_xml)

        in_line_index_xml = doc.createElement("in_line_index")
        in_line_index_xml.appendChild(doc.createTextNode(f"{str(in_line_index)}"))
        object.appendChild(in_line_index_xml)

        orderpack.appendChild(object)

    f = open(save_xml_path, "w")
    # doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='gbk')
    doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
    f.close()


if __name__ == '__main__':
    base_file = "./dataset"
    dataset_file = "./dataset/before_crow"
    save_xml_base_path = "./dataset/Annotations"
    save_jpg_base_path = "./dataset/JPEGImage"
    if not os.path.exists(save_xml_base_path):
        os.makedirs(save_xml_base_path)
    if not os.path.exists(save_jpg_base_path):
        os.makedirs(save_jpg_base_path)
    total_jpg_name = []
    total_json_name = []
    temp_file = os.listdir(dataset_file)
    for i in temp_file:
        if i.endswith(".jpg"):
            total_jpg_name.append(i)
        elif i.endswith(".json"):
            total_json_name.append(i)
    print(len(total_jpg_name))
    print(len(total_json_name))
    # print(total_jpg_name)
    for i in range(len(total_jpg_name)):
        keypoints = []
        save_xml_path = os.path.join(save_xml_base_path, total_jpg_name[i].replace(" ", "_").replace(".jpg", ".xml"))
        save_jpg_path = os.path.join(save_jpg_base_path, total_jpg_name[i].replace(" ", "_"))

        jpg_path = os.path.join(dataset_file, total_jpg_name[i])
        # print(save_xml_path, save_jpg_path)
        img = cv2.imread(jpg_path)
        print(img.shape[0:2])
        if img.shape[0:2] != (480,480):
            continue
        json_path = jpg_path.replace(".jpg", ".json")
        load_f = open(json_path, "r")
        load_dict = json.load(load_f)

        w = load_dict["imageWidth"]
        h = load_dict["imageHeight"]
        h_line = h // 60
        h_ = np.arange(0, h, h_line)
        # h_ = np.arange(0, h)
        line_index = 1
        dst = np.zeros((img.shape[0:2]), dtype=np.uint8)
        for o in load_dict["shapes"]:
            line_points = []
            label_name = o["label"]
            points = o["points"]
            for i in range(len(points) - 1):
                cv2.line(dst, (int(points[i][0]), int(points[i][1])), (int(points[i + 1][0]), int(points[i + 1][1])),
                         color=(255, 0, 0), thickness=1)

        for x in h_:
            temp = dst[x]
            for j in range(len(temp)):
                if temp[j] == 255:
                    # cv2.circle(img, (j, x), radius=1, color=(255, 0, 0), thickness=-1)
                    keypoints.append([int(j), int(x), line_index, 1])

        CreateXML(save_xml_path, keypoints, total_jpg_name[i].replace(" ", "_"), img.shape)
        cv2.imwrite(save_jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        jpg_name = total_jpg_name[i].replace(" ", "_")
        xml_name = jpg_name.replace(".jpg", ".xml")
        # print(jpg_name, xml_name)
        # print(f"Successfully created JPG:{jpg_name} and XML:{xml_name}")
        print("Done!")
