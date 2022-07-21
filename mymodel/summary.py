#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.centernet import centernet

if __name__ == "__main__":
    model = centernet([480, 480, 3], 1, backbone='hrnet')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i, layer.name)