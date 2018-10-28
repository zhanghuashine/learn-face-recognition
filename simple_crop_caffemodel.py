# In face recognition, we just use net to extract features. So the other layers could crop from weight file.
# Cropped caffemodel will match with deply.prototxt.

import caffe

net_file = './model/deploy.prototxt'
caffe_model = './output/weights.caffemodel'

def init_caffe(mode='CPU', device_id=0):
    if 'GPU' == mode or 'gpu' == mode:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    caffe.set_device(device_id)

def crop_caffemodel(net_file, caffe_model):
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    net.save('new.caffemodel')

if __name__ == '__main__':
    init_caffe(mode='GPU', device_id=0)
    crop_caffemodel(net_file, caffe_model)
