from xml.dom.minidom import Document

txt_line = r"C:\Users\Zz\Desktop\centernet-tf2-main\VOCdevkit/VOC2007/JPEGImages/000001.jpg 48,240,195,371,11 8,12,352,498,14"
filename = "000001.jpg"


class XmlMaker:

    def __init__(self, filename):
        self.filename = filename
        self.xmlpath = self.filename.replace(".jpg", ".xml")

    def makexml(self):
        doc = Document()
        orderpack = doc.createElement("annotation")
        doc.appendChild(orderpack)
        folder = doc.createElement("folder")
        folder.appendChild(doc.createTextNode("VOC2007"))
        orderpack.appendChild(folder)

        filename_xml = doc.createElement("filename")
        filename_xml.appendChild(doc.createTextNode(f"{self.filename}"))
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
        flickrid.appendChild(doc.createTextNode(f"341012865"))
        source.appendChild(flickrid)
        orderpack.appendChild(source)

        owner = doc.createElement("owner")
        flickrid_ = doc.createElement("flickrid")
        flickrid_.appendChild(doc.createTextNode(f"Fried Camels"))
        owner.appendChild(flickrid_)

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(f"Jinky the Fruit Bat"))
        owner.appendChild(name)
        orderpack.appendChild(owner)

        size = doc.createElement("size")
        width = doc.createElement("width")
        width.appendChild(doc.createTextNode(f"353"))
        size.appendChild(width)

        height = doc.createElement("height")
        height.appendChild(doc.createTextNode(f"500"))
        size.appendChild(height)

        depth = doc.createElement("depth")
        depth.appendChild(doc.createTextNode(f"3"))
        size.appendChild(depth)
        orderpack.appendChild(size)

        segmented = doc.createElement("segmented")
        segmented.appendChild(doc.createTextNode(f"0"))
        orderpack.appendChild(segmented)

        object = doc.createElement("object")

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(f"dog"))
        object.appendChild(name)

        pose = doc.createElement("pose")
        pose.appendChild(doc.createTextNode(f"Left"))
        object.appendChild(pose)

        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode(f"1"))
        object.appendChild(truncated)

        difficult = doc.createElement("difficult")
        difficult.appendChild(doc.createTextNode(f"0"))
        object.appendChild(difficult)

        bndbox = doc.createElement("bndbox")

        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(f"48"))
        bndbox.appendChild(xmin)

        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(f"240"))
        bndbox.appendChild(ymin)

        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(f"195"))
        bndbox.appendChild(xmax)

        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(f"371"))
        bndbox.appendChild(ymax)

        object.appendChild(bndbox)
        orderpack.appendChild(object)

        object = doc.createElement("object")

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(f"person"))
        object.appendChild(name)

        pose = doc.createElement("pose")
        pose.appendChild(doc.createTextNode(f"Left"))
        object.appendChild(pose)

        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode(f"1"))
        object.appendChild(truncated)

        difficult = doc.createElement("difficult")
        difficult.appendChild(doc.createTextNode(f"0"))
        object.appendChild(difficult)

        bndbox = doc.createElement("bndbox")

        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(f"8"))
        bndbox.appendChild(xmin)

        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(f"12"))
        bndbox.appendChild(ymin)

        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(f"352"))
        bndbox.appendChild(xmax)

        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(f"498"))
        bndbox.appendChild(ymax)

        object.appendChild(bndbox)
        orderpack.appendChild(object)

        f = open(self.xmlpath, "w")
        # doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='gbk')
        doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='gbk')
        f.close()


if __name__ == '__main__':
    read = XmlMaker(filename)
    read.makexml()
