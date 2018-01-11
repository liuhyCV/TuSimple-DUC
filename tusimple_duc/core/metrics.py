import logging
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric


class CompositeEvalMetric(EvalMetric):
    """Manage multiple evaluation metrics."""

    def __init__(self, **kwargs):
        super(CompositeEvalMetric, self).__init__('composite')
        try:
            self.metrics = kwargs['metrics']
        except KeyError:
            self.metrics = []

    def add(self, metric):
        self.metrics.append(metric)

    def get_metric(self, index):
        try:
            return self.metrics[index]
        except IndexError:
            return ValueError("Metric index {} is out of range 0 and {}".format(
                index, len(self.metrics)))

    def update(self, labels, preds):
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def get(self):
        names = []
        results = []
        for metric in self.metrics:
            result = metric.get()
            names.append(result[0])
            results.append(result[1])
        return names, results

    def print_log(self):
        names, results = self.get()
        logging.info('; '.join(['{}: {}'.format(name, val) for name, val in zip(names, results)]))


def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))


class AccWithIgnoreMetric(EvalMetric):
    def __init__(self, ignore_label, name='AccWithIgnore'):
        super(AccWithIgnoreMetric, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._iter_size = 200
        self._nomin_buffer = []
        self._denom_buffer = []

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat) - (label.flat == self._ignore_label).sum()


class IoUMetric(EvalMetric):
    def __init__(self, ignore_label, label_num, name='IoU'):
        self._ignore_label = ignore_label
        self._label_num = label_num
        super(IoUMetric, self).__init__(name=name)

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):
            pred_label = mx.ndarray.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            check_label_shapes(label, pred_label)

            iou = 0
            eps = 1e-6
            # skip_label_num = 0
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1

'''
class SoftmaxLoss(EvalMetric):
    def __init__(self, ignore_label, label_num, name='OverallSoftmaxLoss'):
        super(SoftmaxLoss, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._label_num = label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        loss = 0.0
        cnt = 0.0
        eps = 1e-6
        for i in range(len(labels)):
            prediction = preds[i].asnumpy()[:]
            shape = prediction.shape
            if len(shape) == 4:
                shape = (shape[0], shape[1], shape[2]*shape[3])
                prediction = prediction.reshape(shape)
            label = labels[i].asnumpy()
            soft_label = np.zeros(prediction.shape)
            for b in range(soft_label.shape[0]):
                for c in range(self._label_num):
                    soft_label[b][c][label[b] == c] = 1.0

            loss += (-np.log(prediction[soft_label == 1] + eps)).sum()
            cnt += prediction[soft_label == 1].size
        self.sum_metric += loss
        self.num_inst += cnt
'''

class SoftmaxLoss(EvalMetric):
    def __init__(self, ignore_label, label_num, name='OverallSoftmaxLoss'):
        super(SoftmaxLoss, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._label_num = label_num

    def update(self, labels, preds):
        check_label_shapes(labels, preds)

        loss = 0.0
        cnt = 0.0
        eps = 1e-6
        for i in range(0, 1):
            prediction = preds[i].asnumpy()[:]
            shape = prediction.shape
            print shape
            if len(shape) == 4:
                shape = (shape[0], shape[1], shape[2]*shape[3])
                prediction = prediction.reshape(shape)
            label = labels[i].asnumpy()
            print label.shape
            soft_label = np.zeros(prediction.shape)
            for b in range(soft_label.shape[0]):
                for c in range(self._label_num):
                    soft_label[b][c][label[b] == c] = 1.0

            loss += (-np.log(prediction[soft_label == 1] + eps)).sum()
            cnt += prediction[soft_label == 1].size
        self.sum_metric += loss
        self.num_inst += cnt


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

class InstanceLossMetric(EvalMetric):
    def __init__(self, ignore_label, name='InstanceLoss'):
        super(InstanceLossMetric, self).__init__(name=name)
        self._ignore_label = ignore_label
        self._iter_size = 200
        self._nomin_buffer = []
        self._denom_buffer = []


    def update(self, labels, preds):
        check_label_shapes(labels, preds)
        for i in range(len(labels)):

            in_data_np = preds[i].asnumpy()
            label_data_np = labels[i].asnumpy()

            check_label_shapes(in_data_np, label_data_np)

            in_data = [mx.nd.array(preds[i].asnumpy()), mx.nd.array(labels[i].asnumpy())]
            in_data_np = in_data[0].asnumpy()
            in_shape = in_data_np.shape

            label_data_np = in_data[1].asnumpy()
            label_shape = label_data_np.shape

            batch_size = in_shape[0]
            dim_embedding = in_shape[1]
            feat_height = in_shape[2]
            feat_width = in_shape[3]

            loss = mx.nd.zeros((1))

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

            loss = loss / batch_size


            self.sum_metric += loss.asnumpy()
            self.num_inst += 1