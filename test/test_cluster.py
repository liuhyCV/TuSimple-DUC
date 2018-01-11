import cv2
import sys
import mxnet as mx

print mx.__version__
import argparse

import logging
import os
import numpy as np
import  labels
import time
import re

import layer
from sklearn.cluster import Birch,KMeans
from sklearn.preprocessing import normalize
import random

share_weight_list = ['4_2', '4_3', '4_4', '4_6', '4_7', '4_8', '4_10', '4_11','4_12',
                    '4_14', '4_15', '4_16','4_18', '4_19', '4_20', '4_22', '4_23', '5_2', '5_3']

img_shape = (1024, 2048)


def get_data(img_path, flip, img_shape):
    img = cv2.imread(img_path)
    #img = cv2.flip(img, flip)
    img = cv2.resize(img, img_shape, interpolation=cv2.INTER_NEAREST)
    img = np.array(img, dtype=np.float32)
    # rgb 122.675 116.669 104.008
    #mean = np.array([122.675, 116.669, 104.008])
#    mean = np.array([104.008, 116.669, 122.675])
    mean = np.array([117, 117, 117])
    reshaped_mean = mean.reshape(1, 1, 3)
    reshaped_mean = reshaped_mean[:, :, [2, 1, 0]]

    img = img - reshaped_mean
    img = img[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)
    return img

def load_params(prefix1="./enet",epoch1=140):
    save_dict1 = mx.nd.load('%s-%04d.params' % (prefix1, epoch1))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict1.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
            #temp = re.findall(r"conv(.+?)_3", name)
            #if temp!=[] and temp[0] in share_weight_list:
            #arg_params['dilate1_'+name] = v
            arg_params['dilate'+name] = v
        if tp == 'aux':
            aux_params[name] = v
            #temp = re.findall(r"conv(.+?)_3", name)
            #if temp!=[] and temp[0] in share_weight_list:
             #   aux_params['dilate1_'+name] = v
            aux_params['dilate'+name] = v
    return arg_params,aux_params

def fasttrainID2labels(pred_label, num_class):
    preds = np.array(pred_label, np.uint8)
    preds_color = np.zeros((preds.shape[0],preds.shape[1],3))
    for i in range(num_class):
        index = pred_label == i
        preds[index] = labels.trainId2label[i].id
        preds_color[index] = labels.trainId2label[i].color
    return preds, preds_color[:, :, [2, 1, 0]]

def trainID2labels(pred_label):
    pred_label = np.array(pred_label, np.uint8)
    height, width = pred_label.shape
    pred_color = np.ndarray((height, width, 3))
    for h in range(height):
        for w in range(width):
            if pred_label[h][w] > 18:
                print 'error!!!!!'
            print 'before',pred_label[h][w]
            pred_color[h][w] = labels.trainId2label[pred_label[h][w]].color
            pred_label[h][w] = labels.trainId2label[pred_label[h][w]].id
            print 'after',pred_label[h][w]
            print 'after',pred_color[h][w]
    pred_color = np.array(pred_color, np.uint8)
    [r, g, b] = cv2.split(pred_color)
    result = cv2.merge([b, g, r])
    return pred_label, result

def rearragne(labels, label_num=19, stride=8, cell_width=2, cell_shape=[1024,2048], result_shape=[1024,2048],test_scale=1):
    rpn_width = stride / cell_width
    test_width = (int(cell_shape[1] * test_scale) / stride) * stride
    test_height = (int(cell_shape[0] * test_scale) / stride) * stride
    feat_width = test_width / stride
    feat_height = test_height / stride
    result_width = result_shape[1]
    result_height = result_shape[0]

    print rpn_width, feat_height, feat_width

    labels = labels.reshape((label_num, rpn_width, rpn_width, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, test_height / cell_width, test_width / cell_width))

    labels = labels[:, :int(result_height * test_scale / cell_width),
             :int(result_width * test_scale / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv2.resize(labels, (result_width, result_height), interpolation=cv2.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])
    return labels

def label_visual(label_keans_cluster, num_class):
    color_map = np.zeros((label_keans_cluster.shape[0], label_keans_cluster.shape[1], 3))

    for i in range(0, num_class):
        color = (random.random(), random.random(), random.random())  # generate a random color
        color_map[:,:,0][label_keans_cluster==i] = color[0]
        color_map[:,:,1][label_keans_cluster==i] = color[1]
        color_map[:,:,2][label_keans_cluster==i] = color[2]

        '''
        for i_x in range(0, label_keans_cluster.shape[0]):
            for j_y in range(0, label_keans_cluster.shape[1]):
                if(label_keans_cluster[i_x,j_y]==i):
                    color_map[i_x, j_y, :] = color
        '''


    return color_map

def eval_IOU(args):

    dirc = '../data/cityscapes/'
    lst = 'val.lst.new'
    model_previx = '../models/Instance_loss_Enetfast_CityScapes/2018_01_09_11:43:05/Instance_loss_Enetfast_CityScapes'
    epoch = 22

    ctx = mx.gpu(0)

    enet, enet_args, enet_auxs = mx.model.load_checkpoint(model_previx, epoch)

    exector_2 = enet.simple_bind(ctx, data=(1, 3, 1024, 2048), instanceLoss_label=(1, 1024, 2048), grad_req="null")
    exector_2.copy_params_from(enet_args, enet_auxs, True)

    lines = file(dirc + lst).read().splitlines()
    for line in lines:

        _, data_img_name, label_img_name = line.strip('\n').split("\t")

        label_name = os.path.join(dirc, label_img_name)
        filename = os.path.join(dirc, data_img_name)

        gt_label = cv2.imread(label_name.replace('labelTrainIds', 'color'))
        img = cv2.imread(filename)

        for flip in range(1):
            img_data = get_data(filename, flip, (int(img_shape[1]), int(img_shape[0])))
            img_data = np.expand_dims(img_data, 0)

        data = mx.nd.array(img_data, ctx)

        t = time.time()
        exector_2.forward(is_train=False, data=data)
        print 'time: ',time.time()-t
        output = exector_2.outputs[0].asnumpy()

        output = np.squeeze(output)
        print 'output.shape', output.shape

        output_feature_map_name = 'instance_loss'
        if not os.path.exists('./visual/' + output_feature_map_name):
            os.makedirs('./visual/' + output_feature_map_name)

        feature_map = output
        height_feature_map = feature_map.shape[1]
        width_feature_map = feature_map.shape[2]

        feature_vector = np.zeros((height_feature_map*width_feature_map, feature_map.shape[0]))
        for x in range(0, width_feature_map):
            for y in range(0, height_feature_map):
                feature_vector[x*height_feature_map+y, :] = feature_map[:, y, x]
        print 'feature vecotr generate done!'

        feature_vector = normalize(feature_vector, axis=1, norm='l2')
        print 'feature vecotr norm done!'

        n_clusters = 50
        keans = KMeans(n_clusters=n_clusters)
        keans.fit(feature_vector)
        print 'kmeans done!'

        label_keans_cluster = np.transpose((keans.labels).reshape(width_feature_map, -1))

        label_keans_cluster = cv2.resize(label_keans_cluster, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        result = label_visual(label_keans_cluster, num_class=n_clusters) * 255
        #result = gray2rgb(label_keans_cluster) * 255

        visual = np.zeros((result.shape[0]*2, result.shape[1]*2,3))
        visual[result.shape[0]:,:result.shape[1],:] = result
        visual[result.shape[0]:,result.shape[1]:,:] = gt_label
        visual[:result.shape[0],:result.shape[1],:] = img

        cv2.imwrite('kmeans_'+output_feature_map_name+'.png',visual)

        break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='enet_depth',
                        help='the prefix of the model')
    parser.add_argument('--epoch', type=int, default=1,
                        help='The epoch choosed to load')
    parser.add_argument('--lst', type=str, default='eval_seg.lst',
                        help='The lst file to load')
    parser.add_argument('--gpu', type=int, default=3,
                        help='choose a gpu')
    parser.add_argument('--dir', type=str, default='./',
                        help='the dirctionary of .lst')
    parser.add_argument('--scales', type=str, default='1',
                        help='scales of test')
    parser.add_argument('--num_class', type=int, default=19,
                        help='the number of classes')
    parser.add_argument('--depth', type=bool, default=True,
                        help='if the inputdata has depth image')
    args = parser.parse_args()
    logging.info(args)
    eval_IOU(args)
