import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU
discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
num_classes = settings.CLASS_COUNT

def conv_net(x, is_training=False):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = slim.conv2d(xs, 4, [128, 1], activation_fn=lrelu, scope='conv1')
    print('conv1: ', conv1)
    pool1 = slim.max_pool2d(conv1, [2, 1], scope='pool1')
    print('pool1: ', pool1)

    #convblock 2
    conv2 = slim.conv2d(pool1, 8, [128, 1], activation_fn=lrelu, scope='conv2')
    print('conv2: ', conv2)
    pool2 = slim.max_pool2d(conv2, [2, 1], scope='pool2')
    print('pool2: ', pool2)

    #convblock 3
    conv3 = slim.conv2d(pool2, 16, [128, 1], activation_fn=lrelu, scope='conv3')
    print('conv3: ', conv3)

    #convblock 4
    with tf.name_scope('d2s'):
        d2s4 = tf.depth_to_space(conv3, 4) #upconv
        d2s4 = tf.reshape(d2s4, shape=[-1, sr, 1, 4])
        print('depth2space: ', d2s4)

    with tf.name_scope('concat'):
        conv4 = tf.concat([d2s4, conv1], 3) # <- unet concat conv1 with conv4 16+4
        print('concat: ', conv4)
    conv4 = slim.conv2d(conv4, 16, [128, 1], activation_fn=lrelu, scope='conv4')
    print('conv4: ', conv4)

    #convblock 5
    conv5 = slim.conv2d(conv4, 256, [1, 1], activation_fn=lrelu, scope='conv5')
    print('conv5: ', conv5)
    #out
    with tf.name_scope('output'):
        out = tf.reshape(conv5, [-1, sr, 256])
        print('out: ', out)

    return out
