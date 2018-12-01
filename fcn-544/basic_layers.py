import caffe
from caffe import layers
from caffe import params


def conv(prev, nout, ks=3, stride=1, pad=1):
	ret = layers.Convolution(prev, kernel_size = ks, stride = stride,
								num_output=nout, pad = pad,
								param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	return ret

def relu(prev):
	ret = layers.ReLu(prev, in_place=True)
	return ret

def max_pooling(prev, ks=2, stride=2):
	ret = layers.Pooling(prev, pool=params.Pooling.MAX, kernel_size=ks, stride=stride)
	return ret

def dropout(prev, ratio = 0.5, in_place = True):
	return layers.Dropout(prev, dropout_ratio=ratio, in_place = in_place)

def deconv(prev, nout, ks=4, stride=2, bias_term=False):
	ret = layers.Deconvolution(prev,
			convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride,bias_term = bias_term),
			param = [dict(lr_mult=0)])
	return ret

def sumup(layer1, layer2):
	ret = layers.Eltwise(layer1, layer2, operation=params.Eltwise.SUM)
	return ret
def softmax(layer1, layer2):
	ret = layers.SoftmaxWithLoss(layer1, layer2,
				loss_param=dict(normalize=False, ignore_label=255))
	return ret