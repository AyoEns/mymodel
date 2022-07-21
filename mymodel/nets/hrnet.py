import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Input, Lambda,
                                     Softmax)
from tensorflow.keras.models import Model
from tensorflow.python.keras.initializers import RandomNormal, Constant
# from tensorflow.python.keras.initializers.initializers_v1 import RandomNormal
# from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Reshape, Conv1D, multiply, Conv2DTranspose
from tensorflow.python.keras.regularizers import l2

from nets.backbone import HRnet_Backbone, UpsampleLike

def eca_block(input_feature, b=1, gamma=2, name=""):
    channel = input_feature.shape[3]
    # 根据公式确定一维卷积核的大小
    kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
    # 全局平均池化
    avg_pool = GlobalAveragePooling2D()(input_feature)
    # 打平特征层
    x = Reshape((-1, 1))(avg_pool)
    # ECA中一维卷积层
    x = Conv1D(1, kernel_size=kernel_size, padding="same", name="eca_layer_" + str(name), use_bias=False, )(x)
    # sigmoid函数获得各通道权重值
    x = Activation('sigmoid')(x)
    # 权重值shape变回与原通道一对一对应
    x = Reshape((1, 1, -1))(x)
    # 原输出与权重值相乘，获得具有加强注意力之后的特征层
    output = multiply([input_feature, x])
    return output

def HRnet(image_input, num_classes=21, backbone="hrnetv2_w18"):
    inputs          = image_input
    x, num_filters  = HRnet_Backbone(inputs, backbone)

    x0_0 = x[0]
    x0_1 = UpsampleLike()([x[1], x[0]])
    x0_2 = UpsampleLike()([x[2], x[0]])
    x0_3 = UpsampleLike()([x[3], x[0]])

    x = Concatenate(axis=-1)([x0_0, x0_1, x0_2, x0_3])
    # print(x.shape)
    y1 = Conv2D(128, 3, padding="same", use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer=Constant(0), bias_initializer=Constant(-2.19), activation='sigmoid',
                name="Heatmap")(y1)

    # reg header 中心点偏移量
    y3 = Conv2D(128, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02), name="Centernet_points")(y3)

    return y1, y3
