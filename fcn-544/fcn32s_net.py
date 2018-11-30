import caffe
from caffe import layers as L
from caffe import params as P
from caffe.coord_map import crop
from basic_layers import conv, relu, max_pooling, dropout, deconv, sumup, softmax


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def fcn(mode):
    n = caffe.NetSpec()
    data_params = dict(mode = mode, mean=(104.00699, 116.66877, 122.67892),
                        seed = 1337)
    if mode == 'train':
        data_params['data_dir']='/jet/prs/workspace/VOCdevkit/VOC2012'  ##TODO
        data_layer = 'TrainingDataLayer'
    elif mode == 'test':
        data_params['data_dir']='..' ##TODO
        data_layer = 'TestingDataLayer' 
    else:
        data_params['data_dir']='..'
        data_layer = 'ValidatingDataLayer'
    
    n.data, n.label = layers.Python(module='data_layer', layer = data_layer, ntop=2, param_str = str(data_params))



    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=64, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))



    ########################################################################
    # # layer1 , conv+relu -> conv+relu -> max_pooling
    # net.conv1_1 = conv(net.data, 64, pad=100)
    # net.relu1_1 = relu(net.conv1_1)
    # net.conv1_2 = conv(net.relu1_1, 64)
    # net.relu1_2 = relu(net.conv1_2)
    # net.pool1 = max_pooling(net.relu1_2)

    # # layer2, conv+relu -> conv+relu -> max_pooling
    # net.conv2_1 = conv(net.pool1, 128)
    # net.relu2_1 = relu(net.conv2_1)
    # net.conv2_2 = conv(net.relu2_1, 128)
    # net.relu2_2 = relu(net.conv2_2)
    # net.pool2 = max_pooling(net.relu2_2)

    # # layer3, conv+relu -> conv+relu -> max_pooling
    # net.conv3_1 = conv(net.pool2, 256)
    # net.relu3_1 = relu(net.conv3_1)
    # net.conv3_2 = conv(net.relu3_1, 256)
    # net.relu3_2 = relu(net.conv3_2)
    # net.pool3 = max_pooling(net.relu3_2)

    # # layer4, conv+relu -> conv+relu -> max_pooling
    # net.conv4_1 = conv(net.pool3, 512)
    # net.relu4_1 = relu(net.conv4_1)
    # net.conv4_2 = conv(net.relu4_1, 512)
    # net.relu4_2 = relu(net.conv4_2)
    # net.pool4 = max_pooling(net.relu4_2)

    # # layer5, conv+relu -> conv+relu -> max_pooling
    # net.conv5_1 = conv(net.pool4, 512)
    # net.relu5_1 = relu(net.conv5_1)
    # net.conv5_2 = conv(net.relu5_1, 512)
    # net.relu5_2 = relu(net.conv5_2)
    # net.pool5 = max_pooling(net.relu5_2)


    # # layer6, conv + relu -> dropout
    # net.conv6_1 = conv(net.pool5, 4096, ks=7, pad=0)
    # net.relu6_1 = relu(net.conv6_1)
    # net.drop6_1 = dropout(net.relu6_1)

    # # layer7, conv + relu -> dropout
    # net.conv7_1 = conv(net.drop6_1, 4096, ks=1, pad=0)
    # net.relu7_1 = relu(net.conv7_1)
    # net.drop7_1 = dropout(net.relu7_1)

    # # layer8, forward score
    # net.score1_1 = conv(net.drop7_1, 21, ks=1, pad=0)
    # net.upscore1_1 = deconv(net.score1_1, 21, ks=64, stride = 32)

    # net.score = crop(net.upscore1_1, net.data)
    # net.loss = softmax(net.score, net.data)



    # layer9, skip with layer4: conv -> crop -> sum up -> deconv
    #net.score2_1 = conv(net.pool4, 21, ks=1, pad=0)
    #net.score2_1c = crop(net.score2_1, net.upscore1_1)
    #net.sum_score2_1 = sumup(net.upscore1_1, net.score2_1c)
    #net.upscore2_1 = deconv(net.sum_score2_1, 21)

    # layer10, skip with layer3: conv->crop->sum up->deconv
    #net.score3_1 = conv(net.pool3, 21, ks=1, pad=0)
    #net.score3_1c = crop(net.score3_1, net.upscore2_1)
    #net.sum_score3_1 = sumup(net.upscore2_1, net.score3_1c)
    #net.upscore3_1 = deconv(net.sum_score3_1, 21)

    #net.score = crop(net.upscore3_1, net.data)
    #net.loss = softmax(net.score, net.data)

    return n.to_proto()

def build():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    #with open('test.prototxt', 'w') as f:
        #f.write(str(fcn('test')))

    #with open('val.prototxt', 'w') as f:
        #f.write(str(fcn('val')))

if __name__ == '__main__':
    build()











