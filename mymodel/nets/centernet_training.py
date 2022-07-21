import math
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import Session, InteractiveSession
result = True


def return_True():
    result = True

def return_False():
    result = False


@tf.function
def get_keypoint_num(x_list, batch_size = 4):
    b = batch_size
    num = x_list.shape[1]
    x = []
    x_sum = []
    for i in range(b):
        temp = []
        for j in range(num):
            temp_ = int(0)
            # for k in range(len(x_list[i][j])):
            for k in range(4):
                if tf.where(x_list[i][j][k] == 0, False, True):
                    temp_ += int(1)
            temp_ += int(1)
            temp.append(temp_)
        x.append(temp)
    x_sum = tf.reduce_sum(x, axis=1)
    x = tf.reshape(x, [batch_size, 5])
    return x, x_sum

def get_temp_arrary(i, j, temp_, x, start_index):
    temp_arrary = np.zeros(5)
    end_ = (x[i][j])
    temp__ = tf.slice(input_=temp_, begin=[start_index], size=[x[i][j]])
    temp__ = tf.sort(temp__, direction="ASCENDING")
    temp_arrary[:end_.numpy()] = temp__
    return temp_arrary

def get_x_pre_dis_(batch_size, x_pre_list):
    x_pre_dis = []
    x_temp = 0
    for i in range(batch_size):
        for j in range(x_pre_list.shape[1]):
            x_ = np.zeros((4))
            for k in range(x_pre_list.shape[2] - 1):
                if x_pre_list[i][j][k].numpy() != 0 and k == 0:
                    x_temp = x_pre_list[i][j][k]
                elif x_pre_list[i][j][k].numpy() !=0:
                    x_dis_temp = x_pre_list[i][j][k] - x_temp
                    x_temp = x_pre_list[i][j][k]
                    x_[k - 1] = x_dis_temp
                else:
                    continue
            x_pre_dis.append(x_)

    x_pre_dis = np.array(x_pre_dis).reshape([batch_size, -1])

    return x_pre_dis

def distance_loss(hm, x_list, max_objects=100, batch_size = 4):
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)
    indices = tf.sort(indices, direction="DESCENDING")
    xs = indices // c % w
    ys = indices // c // w
    temp = []
    x, x_num = get_keypoint_num(x_list=x_list, batch_size = batch_size)
    for i in range(batch_size):
        indices_temp = indices[i][:x_num[i]]
        temp.append(indices_temp)
    indices = ys * w + xs
    x_pre_list = []
    x = tf.cast(x, dtype=tf.int32)
    for i in range(batch_size):
        temp_ = tf.sort(temp[i], direction="ASCENDING")
        temp_ = temp_ // c % w
        start_index = 0
        for j in range(x.shape[1]):
            arrary_temp = np.zeros(5)
            flag = tf.cond(x[i][j] == 0, return_True, return_False)
            if result == True:
                arrary_temp = tf.py_function(get_temp_arrary, inp=[i, j, temp_, x, start_index], Tout=tf.float32, name=f"{i+1}")
            x_pre_list.append(arrary_temp)
            start_index = tf.add(x[i][j], start_index)


    x_pre_list = tf.reshape(x_pre_list, (tf.shape(x_list)[0], 5, 5))
    x_pre_list = tf.reshape(x_pre_list, [batch_size,5,5])
    x_pre_dis = tf.py_function(get_x_pre_dis_, inp=[batch_size, x_pre_list], Tout=tf.float32)
    loss_ = tf.keras.losses.MeanSquaredError()
    x_pre_dis = tf.reshape(x_pre_dis, [batch_size, -1])
    x_list = tf.reshape(x_list, [batch_size, -1])
    total_loss = tf.reduce_sum(tf.abs(x_list - x_pre_dis))
    x_num = tf.cast(x_num, tf.float32)
    reg_loss = total_loss / (tf.reduce_sum(x_num) + 1e-4)
    return reg_loss

def focal_loss(hm_pred, hm_true):
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss


def reg_l1_loss(y_pred, y_true, indices, mask):
    #-------------------------------------------------------------------------#
    #   获得batch_size和num_classes
    #-------------------------------------------------------------------------#
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    k = tf.shape(indices)[1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    length = tf.shape(y_pred)[1]
    indices = tf.cast(indices, tf.int32)

    #-------------------------------------------------------------------------#
    #   利用序号取出预测结果中，和真实框相同的特征点的部分
    #-------------------------------------------------------------------------#
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
                    tf.reshape(indices, [-1]))
    y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    #-------------------------------------------------------------------------#
    #   求取l1损失值
    #-------------------------------------------------------------------------#
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def loss(args):
    #-----------------------------------------------------------------------------------------------------------------#
    #   hm_pred：热力图的预测值          (batch_size, 128, 128, num_classes)
    #   reg_pred：中心坐标偏移预测值      (batch_size, 128, 128, 2)
    #   hm_true：热力图的真实值          (batch_size, 128, 128, num_classes)
    #   reg_true：中心坐标偏移真实值          (batch_size, max_objects, 2)
    #   reg_mask：真实值的mask               (batch_size, max_objects)
    #   indices：真实值对应的坐标             (batch_size, max_objects
    #   x_list:横向下，各个点的距离值  (batch_size,any)
    #   y_list:纵向下，各个点的距离值  (batch_size,any)
    #-----------------------------------------------------------------------------------------------------------------#
    # hm_pred, reg_pred, hm_true, reg_true, reg_mask, indices, x_list, y_list = args
    hm_pred, reg_pred, hm_true, reg_true, reg_mask, indices = args
    # hm_pred, reg_pred, hm_true, reg_true, reg_mask, indices = args
    hm_loss_1 = focal_loss(hm_pred, hm_true)
    # mse_loss = tf.keras.losses.MeanSquaredError()
    # hm_loss_1 = mse_loss(hm_true, hm_pred)
    # hm_loss_2 = mse_loss(hm_big_true, hm_big_pred)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss_1 + reg_loss
    # tf.print(total_loss, [hm_loss_1, reg_loss])
    return total_loss


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

