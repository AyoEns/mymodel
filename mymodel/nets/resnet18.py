#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                          Conv2DTranspose, Dropout, MaxPooling2D,
                          ZeroPadding2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape)
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda, Concatenate, multiply, Add, DepthwiseConv2D, PReLU, Subtract
from tensorflow.keras import backend as K

def channel_attention(input_feature, ratio=8, name=""):
    channel = tf.keras.backend.int_shape(input_feature)[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_one_" + str(name))
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_two_" + str(name))

    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    avg_pool = Reshape((1, 1, channel))(avg_pool)
    max_pool = Reshape((1, 1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature, name=""):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name="spatial_attention_" + str(name))(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, name=""):
    cbam_feature = channel_attention(cbam_feature, ratio, name=name)
    cbam_feature = spatial_attention(cbam_feature, name=name)
    return cbam_feature


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02),
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer=RandomNormal(stddev=0.02),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_initializer=RandomNormal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=RandomNormal(stddev=0.02),
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = "same"
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

        # 如果需要激活函数
    if not depth_activation:
        x = Activation('relu')(x)

        # 分离卷积，首先3x3分离卷积，再1x1卷积
        # 3x3采用膨胀卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 1x1卷积，进行压缩
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    return x

def aspp(x,atrous_rates,num_filters=1024):
    size_before = tf.keras.backend.int_shape(x)
    # 分支0
    b0 = Conv2D(num_filters, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # 分支1 rate = 6 (12)
    b1 = SepConv_BN(x, num_filters, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # 分支2 rate = 12 (24)
    b2 = SepConv_BN(x, num_filters, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # 分支3 rate = 18 (36)
    b3 = SepConv_BN(x, num_filters, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # 分支4 全部求平均后，再利用expand_dims扩充维度，之后利用1x1卷积调整通道
    b4 = GlobalAveragePooling2D()(x)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(num_filters, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # 直接利用resize_images扩充hw
    b4 = Lambda(lambda x: tf.compat.v1.image.resize_images(x, size_before[1:3], align_corners=True))(b4)

    # -----------------------------------------#
    #   将五个分支的内容堆叠起来
    #   然后1x1卷积整合特征。
    # -----------------------------------------#
    x = Concatenate()([b4, b0, b1, b2, b3])
    x = Conv2D(num_filters, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    return x


def ResNet18(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=RandomNormal(stddev=0.02), name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = cbam_block(x, name="1")

    # 128,128,64 -> 128,128,256
    x = conv_block(x, 3, [64, 64, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 128], stage=2, block='b')
    x = cbam_block(x, name="2")

    # 128,128,256 -> 64,64,512
    x = conv_block(x, 3, [128, 128, 256], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 256], stage=3, block='b')
    x = cbam_block(x, name="3")

    # 64,64,512 -> 32,32,1024
    x = conv_block(x, 3, [256, 256, 512], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=4, block='b')
    x = cbam_block(x, name="4")
    # 32,32,1024 -> 16,16,2048
    # x = conv_block(x, 3, [512, 512, 1024], stage=5, block='a')
    # x = identity_block(x, 3, [512, 512, 1024], stage=5, block='b')
    return x

def up_projection(lt_, nf):
    ht = Conv2DTranspose(nf, 2, strides=2)(lt_)
    ht = PReLU()(ht)
    lt = ZeroPadding2D(2)(ht)
    lt = Conv2D(nf, 6, 2)(lt)
    lt = PReLU()(lt)
    et = Subtract()([lt, lt_])
    ht1 = Conv2DTranspose(nf, 2, strides=2)(et)
    ht1 = PReLU()(ht1)
    ht1 = Add()([ht, ht1])
    return (ht1)

def centernet_head_resnet18(x,num_classes):
    x = Dropout(rate=0.5)(x)
    #-------------------------------#
    #   解码器
    #-------------------------------#
    num_filters = 256
    # 16, 16, 1024  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    x = aspp(x, atrous_rates=[6, 12, 18], num_filters=512)

    for i in range(3):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # 最终获得128,128,64的特征层
    # hm header 热力图
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer=Constant(0), bias_initializer=Constant(-2.19), activation='sigmoid', name="heatmap")(y1)

    # reg header 中心点偏移量
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02), name="Centernet_points")(y3)
    return y1, y3

if __name__ == '__main__':
    input = Input([512,512,20])
    model = Model(input, ResNet18(input))
    model.summary()