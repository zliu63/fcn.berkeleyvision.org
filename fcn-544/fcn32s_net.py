import caffe
from caffe import layers
from caffe import params
from caffe.coord_map import crop
from basic_layers import conv, relu, max_pooling, dropout, deconv, sumup, softmax




def fcn(mode):
    net = caffe.NetSpec()
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
    
    net.data, net.label = layers.Python(module='data_layer', layer = data_layer, 
                    ntop=2, param_str = str(data_params))

    # layer1 , conv+relu -> conv+relu -> max_pooling
    net.conv1_1 = conv(net.data, 64, pad=100)
    net.relu1_1 = relu(net.conv1_1)
    net.conv1_2 = conv(net.relu1_1, 64)
    net.relu1_2 = relu(net.conv1_2)
    net.pool1 = max_pooling(net.relu1_2)

    # layer2, conv+relu -> conv+relu -> max_pooling
    net.conv2_1 = conv(net.pool1, 128)
    net.relu2_1 = relu(net.conv2_1)
    net.conv2_2 = conv(net.relu2_1, 128)
    net.relu2_2 = relu(net.conv2_2)
    net.pool2 = max_pooling(net.relu2_2)

    # layer3, conv+relu -> conv+relu -> max_pooling
    net.conv3_1 = conv(net.pool2, 256)
    net.relu3_1 = relu(net.conv3_1)
    net.conv3_2 = conv(net.relu3_1, 256)
    net.relu3_2 = relu(net.conv3_2)
    net.conv3_3 = conv(net.relu3_2, 256)
    net.relu3_3 = relu(net.conv3_3)
    net.pool3 = max_pooling(net.relu3_3)

    # layer4, conv+relu -> conv+relu -> max_pooling
    net.conv4_1 = conv(net.pool3, 512)
    net.relu4_1 = relu(net.conv4_1)
    net.conv4_2 = conv(net.relu4_1, 512)
    net.relu4_2 = relu(net.conv4_2)
    net.conv4_3 = conv(net.relu4_2, 512)
    net.relu4_3 = relu(net.conv4_3)
    net.pool4 = max_pooling(net.relu4_3)

    # layer5, conv+relu -> conv+relu -> max_pooling
    net.conv5_1 = conv(net.pool4, 512)
    net.relu5_1 = relu(net.conv5_1)
    net.conv5_2 = conv(net.relu5_1, 512)
    net.relu5_2 = relu(net.conv5_2)
    net.conv5_3 = conv(net.relu5_2, 512)
    net.relu5_3 = relu(net.conv5_3)
    net.pool5 = max_pooling(net.relu5_3)


    # layer6, conv + relu -> dropout
    net.conv6_1 = conv(net.pool5, 4096, ks=7, pad=0)
    net.relu6_1 = relu(net.conv6_1)
    net.drop6 = dropout(net.relu6_1)

    # layer7, conv + relu -> dropout
    net.conv7_1 = conv(net.drop6, 4096, ks=1, pad=0)
    net.relu7_1 = relu(net.conv7_1)
    net.drop7 = dropout(net.relu7_1)

    # layer8, forward score
    net.score1_1 = layers.Convolution(net.drop7, num_output=21, kernel_size=1, pad=0,
                    param = [dict(lr_mult=1,decay_mult=1), dict(lr_mult=2, decay_mult=0)])   #conv(net.drop7, 21, ks=1, pad=0)
    net.upscore1_1 = layers.Deconvolution(net.score1_1, convolution_param=dict(num_output=21,
                    kernel_size=64, stride=32, bias_term=False), param=[dict(lr_mult=0)])  #deconv(net.score1_1, 21, ks=64, stride = 32)

    net.score = crop(net.upscore1_1, net.data)
    net.loss = softmax(net.score, net.label)



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

    return net.to_proto()

def build():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    #with open('test.prototxt', 'w') as f:
        #f.write(str(fcn('test')))

    #with open('val.prototxt', 'w') as f:
        #f.write(str(fcn('val')))

if __name__ == '__main__':
    build()











