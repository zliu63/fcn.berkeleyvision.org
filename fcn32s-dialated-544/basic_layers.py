import caffe
from caffe import layers
from caffe import params


def conv(prev, nout, ks=3, stride=1, pad=1):
    return layers.Convolution(prev, kernel_size = ks, stride = stride,
                                num_output=nout, pad = pad,
                                param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

def atrous(prev, nout, dialation,ks=3, stride=1, pad=1):
    return layers.Convolution(
                            prev,
                            param=[dict(lr_mult=1, decay_mult=1),
                                   dict(lr_mult=2, decay_mult=0)],
                            convolution_param=dict(num_output=nout,
                                                   kernel_size=ks,
                                                   dilation=dialation))


def relu(prev):
    return layers.ReLU(prev, in_place=True)

def max_pooling(prev, ks=2, stride=2):
    return layers.Pooling(prev, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)


def dropout(prev, ratio = 0.5, in_place = True):
    return layers.Dropout(prev, dropout_ratio=ratio, in_place = in_place)

def deconv(prev, nout, ks=4, stride=2, bias_term=False):
    return layers.Deconvolution(prev,
            convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride,bias_term = bias_term),
            param = [dict(lr_mult=0)])

def sumup(layer1, layer2):
    return layers.Eltwise(layer1, layer2, operation=params.Eltwise.SUM)

def softmax(layer1, layer2):
    return layers.SoftmaxWithLoss(layer1, layer2,
                loss_param=dict(normalize=False, ignore_label=255))
