import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU
from functools import partial
import ops

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
embedding_size = settings.EMBEDDING_SIZE
num_classes = settings.CLASS_COUNT

def broadcast(l, emb):
    sh = l.get_shape().as_list()[1]
    emb = emb[:, None, None, :]
    emb = tf.tile(emb, (1,sh,1,1))
    return tf.concat([l, emb], 3)

def GLU(*args, **kwargs):
    kwargs['activation_fn'] = lrelu
    return slim.conv2d(*args, **kwargs)

# Create model
def conv_net(x, cls, dropout, is_training=False):
    xs = tf.expand_dims(x, -1)
    xs = tf.expand_dims(xs, -1)

    conv1 = GLU(xs, 4, [128, 1], scope='conv1_1')
    pool1 = slim.max_pool2d(conv1, [2,1])
    print('conv1', conv1)

    #convblock 2
    conv2 = GLU(pool1, 8, [128, 1], scope='conv2_1')
    conv2 = slim.max_pool2d(conv2, [2,1])
    print('conv2: ', conv2)

    with tf.variable_scope('embedding'):
        emb1 = embedding(cls, embedding_size, num_classes)
        embedded = broadcast(conv2, emb1)
    print('embedded:', embedded)

    #convblock 3
    conv3 = GLU(embedded, 16, [128, 1], scope='conv3_1')
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 2) #upconv
    print('d2sp: ', conv4)
    conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 4]) # reshape upconvolution to have proper shape

    conv4 = GLU(conv4, 16, [128, 1], scope='conv4_1')
    #convblock 5
    conv4 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    print('conv4: ', conv4)


    conv5 = GLU(conv4, discrete_class, [2,1], scope='conv5')
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, sr, discrete_class])
    print('out: ', out)
    return out

