import mxnet as mx
import layer
def get_conv(name, data, num_filter, kernel, stride, pad,
             with_relu, bn_momentum=0.9, dilate=(1, 1)):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        eps=2e-4 # issue of cudnn
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=bn)
        if with_relu else bn
    )
def get_shared_conv(name, data, num_filter, kernel, stride, pad,
             with_relu, weight,bn_momentum=0.9, dilate=(1, 1)):
    conv = mx.symbol.Convolution(
        name=name,
        data=data,
        num_filter=num_filter,
        weight=weight,
        kernel=kernel,
        stride=stride,
        pad=pad,
        dilate=dilate,
        no_bias=True
    )
    bn = mx.symbol.BatchNorm(
        name=name + '_bn',
        data=conv,
        fix_gamma=False,
        momentum=bn_momentum,
        # Same with https://github.com/soumith/cudnn.torch/blob/master/BatchNormalization.lua
        eps=2e-4 # issue of cudnn
    )
    return (
        # It's better to remove ReLU here
        # https://github.com/gcr/torch-residual-networks
        mx.symbol.LeakyReLU(name=name + '_prelu', act_type='prelu', data=bn)
        if with_relu else bn
    )



def initila_block(data):

    conv = get_conv(name='initial_conv1',
                           data=data,
                           num_filter=16,
                           kernel=(3, 3),
                           pad=(1, 1),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(2, 2))

    conv = get_conv(name='initial_conv2',
                           data=conv,
                           num_filter=16,
                           kernel=(3, 3),
                           pad=(1, 1),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(1, 1))

    conv = get_conv(name='initial_conv3',
                           data=conv,
                           num_filter=16,
                           kernel=(3, 3),
                           pad=(1, 1),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(1, 1))

    return conv


def make_shared_block(name, data, num_filter, bn_momentum,
               down_sample=False, up_sample=False,dim_match=False,
               dilated=(1, 1), asymmetric=0,conv_weight=0,share=False):
    """maxpooling & padding"""
    if dim_match or down_sample:
        # 1x1 conv ensures that channel equal to main branch
        maxpool = get_conv(name=name + '_proj_maxpool',
                           data=data,
                           num_filter=num_filter,
                           kernel=(2, 2) if down_sample else (1, 1),
                           pad=(0, 0),
                           with_relu=True,
                           bn_momentum=bn_momentum,
                           stride=(2, 2) if down_sample else (1, 1))

    elif up_sample:
        # maxunpooling.
        maxpool = mx.symbol.Deconvolution(name=name + '_unpooling',
                                   data=data,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))

        # Reference: https://github.com/e-lab/ENet-training/blob/master/train/models/decoder.lua
        # Padding is replaced by 1x1 convolution
        maxpool = get_conv(name=name + '_padding',
                           data=maxpool,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=(1, 1),
                           pad=(0, 0),
                           bn_momentum=bn_momentum,
                           with_relu=False)
    # main branch begin
    proj = get_conv(name=name + '_proj0',
                    data=data,
                    num_filter=num_filter,
                    kernel=(3, 3) if not down_sample else (2, 2),
                    stride=(1, 1) if not down_sample else (2, 2),
                    pad=(1, 1) if not down_sample else (0, 0),
                    with_relu=True,
                    bn_momentum=bn_momentum)

    if up_sample:
        conv = mx.symbol.Deconvolution(name=name + '_deconv',
                                   data=proj,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))
    else:
        if asymmetric == 0:

            conv11 = get_shared_conv(name=name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            weight=conv_weight,
                            kernel=(3, 3),
                            pad=dilated,
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)

            conv12 = get_shared_conv(name='dilate' + name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            weight=conv_weight,
                            kernel=(3, 3),
                            pad=(dilated[0]*3/2,dilated[1]*3/2),
                            dilate=(dilated[0]*3/2,dilated[1]*3/2),
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = conv11
            '''
            if dilated==(1, 1) or dilated==(2, 2):
                conv = conv11
            else:
		conv = (conv11 + conv12) / 2
           '''
        else:
            conv = get_conv(name=name + '_conv1',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(1, asymmetric),
                            pad=(0, asymmetric / 2),
                            stride=(1, 1),
                            dilate=dilated,
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = get_conv(name=name + '_conv2',
                            data=conv,
                            num_filter=num_filter,
                            kernel=(asymmetric, 1),
                            pad=(asymmetric / 2, 0),
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
    '''
    regular = mx.symbol.Convolution(name=name + '_expansion',
                                        data=conv,
                                        num_filter=num_filter,
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        stride=(1, 1),
                                        no_bias=True)
    regular = mx.symbol.BatchNorm(
        name=name + '_expansion_bn',
        data=regular,
        fix_gamma=False,
        momentum=bn_momentum,
        eps=2e-4 # issue of cudnn
    )
    regular._set_attr(mirror_stage='True')
    # main branch end
    # TODO: spatial dropout
    '''
    if down_sample or up_sample or dim_match:
        regular = mx.symbol.ElementWiseSum(maxpool, conv, name =  name + "_plus")
    else:
        regular = mx.symbol.ElementWiseSum(data, conv, name =  name + "_plus")

    regular = mx.symbol.LeakyReLU(name=name + '_expansion_prelu', act_type='prelu', data=regular)
    return regular


def make_block(name, data, num_filter, bn_momentum,
               down_sample=False, up_sample=False,
               dilated=(1, 1), asymmetric=0):
    """maxpooling & padding"""
    if down_sample:
        # 1x1 conv ensures that channel equal to main branch
        maxpool = get_conv(name=name + '_proj_maxpool',
                           data=data,
                           num_filter=num_filter,
                           kernel=(2, 2),
                           pad=(0, 0),
                           with_relu=True,
                           bn_momentum=bn_momentum,
                           stride=(2, 2))

    elif up_sample:
        # maxunpooling.
        maxpool = mx.symbol.Deconvolution(name=name + '_unpooling',
                                   data=data,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))

        # Reference: https://github.com/e-lab/ENet-training/blob/master/train/models/decoder.lua
        # Padding is replaced by 1x1 convolution
        maxpool = get_conv(name=name + '_padding',
                           data=maxpool,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=(1, 1),
                           pad=(0, 0),
                           bn_momentum=bn_momentum,
                           with_relu=False)
    # main branch begin
    proj = get_conv(name=name + '_proj0',
                    data=data,
                    num_filter=num_filter,
                    kernel=(3, 3) if not down_sample else (2, 2),
                    stride=(1, 1) if not down_sample else (2, 2),
                    pad=(1, 1) if not down_sample else (0, 0),
                    with_relu=True,
                    bn_momentum=bn_momentum)

    if up_sample:
        conv = mx.symbol.Deconvolution(name=name + '_deconv',
                                   data=proj,
                                   num_filter=num_filter,
                                   kernel=(4, 4),
                                   stride=(2, 2),
                                   pad=(1, 1))
    else:
        if asymmetric == 0:
            conv = get_conv(name=name + '_conv',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(3, 3),
                            pad=dilated,
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
        else:
            conv = get_conv(name=name + '_conv1',
                            data=proj,
                            num_filter=num_filter,
                            kernel=(1, asymmetric),
                            pad=(0, asymmetric / 2),
                            stride=(1, 1),
                            dilate=dilated,
                            with_relu=True,
                            bn_momentum=bn_momentum)
            conv = get_conv(name=name + '_conv2',
                            data=conv,
                            num_filter=num_filter,
                            kernel=(asymmetric, 1),
                            pad=(asymmetric / 2, 0),
                            dilate=dilated,
                            stride=(1, 1),
                            with_relu=True,
                            bn_momentum=bn_momentum)
    '''
    regular = mx.symbol.Convolution(name=name + '_expansion',
                                        data=conv,
                                        num_filter=num_filter,
                                        kernel=(1, 1),
                                        pad=(0, 0),
                                        stride=(1, 1),
                                        no_bias=True)
    regular = mx.symbol.BatchNorm(
        name=name + '_expansion_bn',
        data=regular,
        fix_gamma=False,
        momentum=bn_momentum,
        eps=2e-4 # issue of cudnn
    )
    regular._set_attr(mirror_stage='True')
    # main branch end
    # TODO: spatial dropout
    '''
    if down_sample or up_sample:
        regular = mx.symbol.ElementWiseSum(maxpool, conv, name =  name + "_plus")
    else:
        regular = mx.symbol.ElementWiseSum(data, conv, name =  name + "_plus")

    regular = mx.symbol.LeakyReLU(name=name + '_expansion_prelu', act_type='prelu', data=regular)
    return regular

def shared_comput(data, bn_momentum, name, weight_list):

    num_filter = 64

    data = data0 = make_shared_block(name=name+"bottleneck2.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False,conv_weight=weight_list[0])
    data = make_shared_block(name=name+"bottleneck2.1", data=data, num_filter=num_filter, bn_momentum=bn_momentum,conv_weight=weight_list[1])
    data = make_shared_block(name=name+"bottleneck2.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(1, 1),conv_weight=weight_list[2])
    data = make_shared_block(name=name+"bottleneck2.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5,conv_weight=weight_list[3])
    data = make_shared_block(name=name+"bottleneck2.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2),conv_weight=weight_list[4], share=True)
    data = make_shared_block(name=name+"bottleneck2.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum,conv_weight=weight_list[5])
    data = make_shared_block(name=name+"bottleneck2.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4),conv_weight=weight_list[6], share=True)
    data = make_shared_block(name=name+"bottleneck2.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5,conv_weight=weight_list[7])
    data = make_shared_block(name=name+"bottleneck2.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8),conv_weight=weight_list[8], share=True)

    data = make_shared_block(name=name+"bottleneck2.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum,conv_weight=weight_list[9])

    data = make_shared_block(name=name+"bottleneck2.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16),conv_weight=weight_list[10], share=True)
    data0 = make_block(name=name+"projection2", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    data._set_attr(mirror_stage='True')
    ##level 3
    num_filter = 64
    data = data0 = make_shared_block(name=name+"bottleneck3.1", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,conv_weight=weight_list[11])
    data = make_shared_block(name=name+"bottleneck3.2", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(1, 1),conv_weight=weight_list[12], share=False)
    data = make_shared_block(name=name+"bottleneck3.3", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5,conv_weight=weight_list[13], share=False)
    data = make_shared_block(name=name+"bottleneck3.4", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(2, 2),conv_weight=weight_list[14], share=True)
    data = make_shared_block(name=name+"bottleneck3.5", data=data, num_filter=num_filter, bn_momentum=bn_momentum,conv_weight=weight_list[15], share=True)
    data = make_shared_block(name=name+"bottleneck3.6", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(4, 4),conv_weight=weight_list[16], share=True)
    data = make_shared_block(name=name+"bottleneck3.7", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      asymmetric=5,conv_weight=weight_list[17])
    data = make_shared_block(name=name+"bottleneck3.8", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(8, 8),conv_weight=weight_list[18], share=True)
    data = make_shared_block(name=name+"bottleneck3.9", data=data, num_filter=num_filter, bn_momentum=bn_momentum,conv_weight=weight_list[19])

    data = make_shared_block(name=name+"bottleneck3.10", data=data, num_filter=num_filter, bn_momentum=bn_momentum,
                      dilated=(16, 16),conv_weight=weight_list[20], share=True)
    data0 = make_block(name=name+"projection3", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    data._set_attr(mirror_stage='True')

    return data

def get_body(data, num_class, bn_momentum):
    ##level 0

    #get shared-weight Variable
    k = -1
    weight_list = []

    for i in range(22):
        k +=1
        if i<11:
            weight_list.append(mx.symbol.Variable("bottleneck2.{}_conv_weight".format(k)))
        else:
            if i==11:
                k=1
            weight_list.append(mx.symbol.Variable("bottleneck3.{}_conv_weight".format(k)))

    data = initila_block(data)  # 16

    ##level 1
    num_filter = 32
    data = data0 = make_block(name="bottleneck1.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum, down_sample=True, up_sample=False)  # 64
    for block in range(4):
        data = make_block(name='bottleneck1.%d' % (block + 1),
                          data=data, num_filter=num_filter,  bn_momentum=bn_momentum,
                          down_sample=False, up_sample=False)
    data0 = make_block(name="projection1", data=data0, num_filter=num_filter, bn_momentum=bn_momentum)
    data = data + data0
    data._set_attr(mirror_stage='True')
    ##level 2,3

    data = shared_comput(data, bn_momentum, "", weight_list)

    ##level 4
    num_filter = 32
    data  = make_block(name="bottleneck4.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = make_block(name="bottleneck4.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)
    data = make_block(name="bottleneck4.2", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)

    ##level 5
    num_filter = 16
    data = make_block(name="bottleneck5.0", data=data, num_filter=num_filter,
                              bn_momentum=bn_momentum,
                              up_sample=True)
    data = make_block(name="bottleneck5.1", data=data, num_filter=num_filter,  bn_momentum=bn_momentum)

    ##level 6
    #data = mx.symbol.Deconvolution(data=data, kernel=(16, 16), stride=(2, 2), num_filter=num_class,
      #                             name="newfullconv")
    data = mx.symbol.UpSampling(data, num_args=1, scale=2, sample_type='nearest', name='upsample')
    data = get_conv(name='score_conv1',
                           data=data,
                           num_filter=num_class,
                           kernel=(16, 1),
                           pad=(8, 0),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(1, 1))
    data = get_conv(name='score_conv2',
                           data=data,
                           num_filter=num_class,
                           kernel=(1, 16),
			   dilate = (1, 1),
                           pad=(0, 8),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(1, 1))
    data = get_conv(name='score_conv3',
                           data=data,
                           num_filter=num_class,
                           kernel=(3, 3),
			   dilate = (1, 1),
                           pad=(1, 1),
                           with_relu=True,
                           bn_momentum=0.9,
                           stride=(1, 1))
    return data


def get_enet_symbol(num_classes, ohem=False, bn_momentum=0.9, spl=0, dim_embedding=8):
    # data = mx.symbol.Variable(name='data')
    # label_inst = mx.symbol.Variable(name='label_inst')
    # body = get_body(
    #     data,
    #     num_classes,
    #     bn_momentum
    # )
    #
    # embedding_inst = get_embeddings(
    #     data,
    #     dim_embedding,
    #     bn_momentum
    # )
    #
    # body = mx.symbol.Crop(*[body, data], name="fullconv_crop")
    #
    # softmax = mx.symbol.SoftmaxOutput(data=body, multi_output=True, use_ignore=True, ignore_label=255, normalization='valid',
    #                                       name="softmax")  # is_hidden_layer=True,
    #
    # loss_inst = mx.symbol.Custom(data=embedding_inst, label_inst=label_inst, ggggggg=body, name='instanceLoss', op_type='clusterInstanceLoss')


    data = mx.symbol.Variable(name='data')
    embedding_inst = get_body(
        data,
        8,
        bn_momentum
    )

    embedding_inst = mx.symbol.Crop(*[embedding_inst, data], name="fullconv_crop")

    loss_inst = mx.symbol.Custom(data=embedding_inst, name='instanceLoss', op_type='clusterInstanceLoss')

    return loss_inst

