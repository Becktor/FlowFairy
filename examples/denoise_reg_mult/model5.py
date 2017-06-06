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


def conv_net(x, dropout, is_training=False):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = GLU(xs, 4, [256, 1],  scope='conv1_1',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training,
                                   'decay': 0.9})
    pool1 = slim.max_pool2d(conv1, [2, 1])
    print('conv1: ', pool1)

    #convblock 2
    conv2 = GLU(pool1, 16, [64, 1], scope='conv2')
    pool2 = slim.max_pool2d(conv2, [2, 1])
    print('conv2: ', pool2)

    #convblock 3
    conv3 = GLU(pool2, 16, [64, 1], scope='conv3')
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 4) #upconv
    print('d2sp: ', conv4)
    conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 4]) # reshape upconvolution to have proper shape

    conv4 = GLU(conv4, 16, [64, 1], scope='conv4')
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last

    conv5 = GLU(conv5, 256, [1,1], scope='conv5')
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, sr, 256])
    print('out: ', out)
    return out
