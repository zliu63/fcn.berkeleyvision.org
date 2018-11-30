import random
import caffe
import numpy as np
from PIL import Image


class TrainingDataLayer(caffe.Layer):

	def setup(self, bottom, top):
		params = eval(self.param_str)
		self.data_dir = params['data_dir']
		self.mode = params['mode']
		self.mean = np.array(params['mean'])
		self.random = params.get('randominze', True)
		self.seed = params.get('seed', None)
		
        images_file = '{}/ImageSets/Segmentation/{}.txt'.format(self.data_dir, self.mode)
        self.indices = open(images_file, 'r').read().splitlines()
        self.idx = 0

        random.seed(self.seed)
        self.idx = random.randint(0,len(self.indices)-1)

    def reshape(self, bottom, top):
    	self.data = self.load_image(self.indices[self.idx])
    	self.label = self.load_label(self.indices[self.idx])

    	top[0].reshape(1, *self.data.shape)
    	top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
    	top[0].data[...] = self.data
    	top[1].data[...] = self.label

		self.idx = random.randint(0, len(self.indices)-1)

	def backward(self, top, propagate_down):
		pass

	def load_image(self, idx):
		image = Image.open('{}/JPEGImages/{}.jpg'.format(self.data_dir, idx))
		ret = np.array(image, dtype = np.float32)
		ret = ret[:,:,::-1]
		ret -= self.mean
		ret = ret.transpose((2,0,1))
		return ret

	def load_label(self, idx):
		image = Image.open('{}/SegmentationClass/{}.png'.format(self.data_dir, idx))
		label = np.array(image, dtype=np.uint8)
		label = label[np.newaxis, ...]
		return label