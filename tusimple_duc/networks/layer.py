"""
softmax with bootstrapping
"""
import heapq
import logging
from functools import partial

import numpy as np

import mxnet as mx
from mxnet import nd, autograd

import cv2


in_margin = 0.5
out_margin = 1.5
Lnorm = 2

def Instlabel_to_IdsKey(label):
    # Inst label read should use imread(imname, flags=cv2.IMREAD_UNCHANGED)
    inst_label_unique = np.int16(np.unique(label))
    #print 'np.unique(label)', np.unique(label)

    inst_ids_key = {}
    for i in range(0, len(inst_label_unique)):
        i_inst_ids = inst_label_unique[i]
        if (i_inst_ids < 500):
            local_list = []
            local_list.append(i_inst_ids)
            inst_ids_key[i_inst_ids] = local_list
        else:
            if inst_ids_key.has_key(i_inst_ids / 1000):
                inst_ids_key[i_inst_ids / 1000].append(i_inst_ids)
            else:
                local_list = []
                local_list.append(i_inst_ids)
                inst_ids_key[i_inst_ids / 1000] = local_list

    inst_label_unique[inst_label_unique > 100] = inst_label_unique[inst_label_unique > 100] / 1000

    ignore_inst_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 255]

    # only retain ids that hasInstances=True
    # ignore_inst_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 255]

    inst_label_unique = set(inst_label_unique) - set(ignore_inst_ids)
    inst_label_unique = list(inst_label_unique)

    return inst_ids_key, inst_label_unique


class ClusterInstanceLoss(mx.operator.CustomOp):
    def __init__(self, **kwargs):
        super(ClusterInstanceLoss, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        in_data_np = in_data[0].asnumpy()
        in_shape = in_data_np.shape

        label_data_np = in_data[1].asnumpy()
        label_shape = label_data_np.shape

        batch_size = in_shape[0]
        dim_embedding = in_shape[1]
        feat_height = in_shape[2]
        feat_width = in_shape[3]


        in_data[0].attach_grad()

        loss = mx.nd.zeros((1))

        with autograd.record():

            loss_v = mx.nd.zeros((1))
            loss_d = mx.nd.zeros((1))
            loss_r = mx.nd.zeros((1))

            for b in range(0, batch_size):
                # dim * h * w
                local_embedding = mx.nd.slice_axis(in_data[0], axis=0, begin=b, end=b+1)
                # 1 * h * w
                local_label_inst = mx.nd.slice_axis(in_data[1], axis=0, begin=b, end=b+1)

                inst_ids_key, inst_label_unique = Instlabel_to_IdsKey(local_label_inst.asnumpy())
                num_classes = len(inst_label_unique)

                for i_class in range(0, num_classes):

                    # h * w
                    #local_label_class = mx.nd.slice_axis(local_label_inst, axis=0, begin=i_class, end=i_class+1)
                    i_class_inst = inst_ids_key[inst_label_unique[i_class]]
                    num_inst = len(i_class_inst)

                    num_inst = min(num_inst, 10)

                    loss_var_perclass = mx.nd.zeros((1))
                    loss_dist_perclass = mx.nd.zeros((1))

                    for i_inst in range(0, num_inst):
                        # h * w
                        local_mask_inst = local_label_inst==i_class_inst[i_inst]
                        num_cluser_pixs = mx.nd.sum(local_mask_inst)

                        local_mask_inst = local_mask_inst.reshape((1, 1, feat_height, feat_width))

                        local_mask_embedding = local_embedding * local_mask_inst

                        mean = mx.nd.sum(local_mask_embedding, axis=[2,3]) / num_cluser_pixs

                        mean = mean.reshape((1, -1, 1, 1)) * local_mask_inst

                        # ||MUc - xi|| - DETAv
                        # h * w
                        dist_in = mx.nd.sqrt(mx.nd.sum(mx.nd.square(local_mask_embedding - mean), axis=0)) - in_margin
                        local_var = mx.nd.sum(mx.nd.square(mx.nd.where(dist_in>0, dist_in, mx.nd.zeros(dist_in.shape))))

                        loss_var_perclass = loss_var_perclass + local_var / num_cluser_pixs

                    loss_var_perclass = loss_var_perclass / num_inst

                    if (num_inst>1):

                        for i_inst in range(0, num_inst):
                            for j_inst in range(i_inst+1, num_inst):
                                # i_inst mean
                                local_mask_inst = local_label_inst == i_class_inst[i_inst]
                                local_mask_inst = local_mask_inst.reshape((1, 1, feat_height, feat_width))

                                local_mask_embedding = local_embedding * local_mask_inst
                                # i_mean shape: dim
                                i_mean = mx.nd.sum(local_mask_embedding, axis=[2, 3]) / mx.nd.sum(local_mask_inst)

                                # j_inst mean
                                local_mask_inst = local_label_inst == i_class_inst[j_inst]
                                local_mask_inst = local_mask_inst.reshape((1, 1, feat_height, feat_width))
                                local_mask_embedding = local_embedding * local_mask_inst
                                j_mean = mx.nd.sum(local_mask_embedding, axis=[2, 3]) / mx.nd.sum(local_mask_inst)

                                dist_out = 2 * out_margin - mx.nd.sqrt(mx.nd.sum(mx.nd.square(i_mean - j_mean)))
                                local_dist = mx.nd.where(dist_out>0, dist_out, mx.nd.zeros(dist_out.shape))

                                loss_dist_perclass = loss_dist_perclass + local_dist

                        loss_dist_perclass = loss_dist_perclass / (num_inst*(num_inst-1))

                    for i_inst in range(0, num_inst):
                        # i_inst mean
                        local_mask_inst = local_label_inst == i_class_inst[i_inst]
                        local_mask_inst = local_mask_inst.reshape((1, 1, feat_height, feat_width))

                        local_mask_embedding = local_embedding * local_mask_inst
                        # i_mean shape: dim
                        i_mean = mx.nd.sum(local_mask_embedding, axis=[2, 3]) / mx.nd.sum(local_mask_inst)

                        loss_r = loss_r + mx.nd.sqrt(mx.nd.sum(mx.nd.square(i_mean)))

                    loss_v = loss_v + loss_var_perclass
                    loss_d = loss_d + loss_dist_perclass

                    loss = loss + loss_v + loss_d + 0.001 * loss_r / num_inst
                    # loss = loss + loss_var_perclass

            loss = loss / batch_size
            print num_inst
            print loss_v, loss_d, loss_r

        loss.backward()

        self.assign(in_grad[0], req[0], in_data[0].grad)



@mx.operator.register('clusterInstanceLoss')
class ClusterInstanceLossProp(mx.operator.CustomOpProp):
    def __init__(self, **kwargs):
        super(ClusterInstanceLossProp, self).__init__(need_top_grad=False)
        self._kwargs = kwargs.copy()

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_inst_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, label_inst_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return ClusterInstanceLoss(**self._kwargs)
