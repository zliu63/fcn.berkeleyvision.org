import caffe
import numpy as np
import os

import score

# 2D bilinear kernel for upsampling
def upsample(s):
	up_f = (s + 1) // 2
	center = up_f - 1 if s % 2 else up_f - 0.5
    x, y = np.ogrid[:size, :size]
    return (1 - abs(x - center) / up_f) * (1 - abs(y - center) / up_f)

# Set weights for bilinear kernels
def interpolation(net, layers):
	for l in layers:
		in_s, out_s, h, w = net.params[l][0].data.shape
		if in_s != out_s and out_s != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filter = upsample(h)
        net.params[l][0].data[range(m), range(k), :, :] = filter

# try:
#     import setproctitle
#     setproctitle.setproctitle(os.path.basename(os.getcwd()))
# except:
#     pass

caffe.set_device(0)
caffe.set_mode_gpu()

weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# setup bilinear kernels
interpolation_layers = [k for k in solver.net.params.keys() if 'up' in k]
interpolation(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/segvalid11.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')




