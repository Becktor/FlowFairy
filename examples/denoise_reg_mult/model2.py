import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
falearning_rate = settings.LEARNING_RATE
num_classes = settings.CLASS_COUNT

def conv_net(x,  dropout, is_training=False):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = slim.conv2d(xs, 4, [128, 1], activation_fn=lrelu, scope='conv1')
    print('conv1: ', conv1)
    pool1 = slim.max_pool2d(conv1, [2, 1], scope='pool1')
    print('pool1: ', pool1)
    #convblock 2
    conv2 = slim.conv2d(pool1, 8, [64, 1], activation_fn=lrelu, scope='conv2')
    print('conv2: ', conv2)
    pool2 = slim.max_pool2d(conv2, [2, 1], scope='pool2')
    print('pool2: ', pool2)
    #upconvolution
    upconv1 = slim.convolution2d_transpose(pool2, 8, [64,1], [2,1], scope='upconv1')
    print('upconv1: ',upconv1)
    #upconvolution
    upconv2 = slim.convolution2d_transpose(upconv1, 8, [64,1], [2,1], scope='upconv2')
    print('upconv1: ',upconv2)
    #convblock 3
    conv3 = slim.conv2d(upconv2, 1, [64, 1], activation_fn=lrelu, scope='conv3')
    print('conv3: ', conv3)
    #out
    out = tf.reshape(conv3, [-1, sr])
    print('out: ', out)
    return out
