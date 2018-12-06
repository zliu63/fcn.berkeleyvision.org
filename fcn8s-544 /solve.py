import caffe
import numpy as np
import os

import score

# 2D bilinear kernel for upsampling
def upsample(s):
	up_f = (s + 1) // 2
	center = up_f - 1 if s % 2 else up_f - 0.5
	x, y = np.ogrid[:s, :s]
	return (1 - abs(x - center) / up_f) * (1 - abs(y - center) / up_f)

# Set weights for bilinear kernel
def interpolation(net, layers):
	for l in layers:
		in_s, out_s, h, w = net.params[l][0].data.shape
		if in_s != out_s and out_s != 1:
			print('input + output channels need to be the same or |output| == 1')
			raise
		if h != w:
			print('filters need to be square')
			raise
		filter = upsample(h)
		net.params[l][0].data[range(in_s), range(out_s), :, :] = filter

def transplant(new_net, net, suffix=''):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.

    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print('dropping', p)
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print('dropping', p, i)
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print('coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape)
            else:
                print('copying', p, ' -> ', p_new, i)
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat
# try:
#     import setproctitle
#     setproctitle.setproctitle(os.path.basename(os.getcwd()))
# except:
#     pass

caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../models-544/fcn16s.caffemodel'
#model = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

#base_net = caffe.Net(model, weights, caffe.TEST)
#transplant(solver.net, base_net)
#del base_net

# setup bilinear kernels
interpolation_layers = [k for k in solver.net.params.keys() if 'up' in k]
interpolation(solver.net, interpolation_layers)

# scoring
val = np.loadtxt('val.txt', dtype=str)

for _ in range(5):
	solver.step(20000)
	score.seg_tests(solver, False, val, layer='score')




