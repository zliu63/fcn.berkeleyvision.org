import caffe
from caffe import layers
from caffe import params


def conv(bottom, nout, ks=3, stride=1, pad=1):
	return layers.Convolution(bottom, kernel_size = ks, stride = stride,
								num_output=nout, pad = pad,
								param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])


def relu(bottom):
	return layers.ReLU(bottom, in_place=True)

def max_pooling(bottom, ks=2, stride=2):
	return layers.Pooling(bottom, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)


def dropout(bottom, ratio = 0.5, in_place = True):
	return layers.Dropout(bottom, dropout_ratio=ratio, in_place = in_place)

def deconv(bottom, nout, ks=4, stride=2, bias_term=False):
	return layers.Deconvolution(bottom,
			convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride,bias_term = bias_term),
			param = [dict(lr_mult=0)])

def sumup(layer1, layer2):
	return layers.Eltwise(layer1, layer2, operation=params.Eltwise.SUM)

def softmax(layer1, layer2):
	return layers.SoftmaxWithLoss(layer1, layer2,
				loss_param=dict(normalize=False, ignore_label=255))
