import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
import util
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU, dense_block
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
    print(sh)
    emb = tf.tile(emb, (1,sh,1,1))
    return tf.concat([l, emb], 3)


def dense(inl, depth, pool=True, is_training=False):
    return dense_block(inl, GLU, {'num_filters':depth, 'kernel_size':[32,1]}, pool=pool, is_training=is_training)

def trdense(tr):
    def wrp(*args, **kwargs):
        return dense(*args, **kwargs, is_training=tr)
    return wrp

# Create model
def conv_net(x, cls, dropout, is_training=False):
    util.reset()
    dense = trdense(is_training)
    xs = tf.expand_dims(x, -2)

    conv1 = GLU(xs, 4, [128, 1], scope='conv1_1', normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training})
    print('conv1', conv1)

    conv1_d1 = GLU(conv1, 4, [128, 1], scope='conv1_d1')
    print('conv1_d1 ', conv1_d1)

    # Parallel
    conv1_d2 = GLU(conv1, 4, [128, 1], rate=2, scope='conv1_d2')
    print('conv1_d2 ', conv1_d2)

    conv1_d4 = GLU(conv1, 4, [128, 1], rate=4, scope='conv1_d4')
    print('conv1_d4 ', conv1_d4)

    conv1 = tf.concat([conv1_d1, conv1_d2, conv1_d4], 3)
    print('conv1_concat', conv1)

    d1p, d1 = dense(conv1, 8)
    d1s = d1.get_shape().as_list()
    print('d1', d1)

    d2p, d2 = dense(d1p, 16)
    d2s = d2.get_shape().as_list()
    print('d2', d2)

    _, d3 = dense(d2p, 32, pool=False)
    print('d3', d3)

    #convblock 4
    d4 = tf.depth_to_space(d3, 2) #upconv
    d4 = tf.reshape(d4, shape=d2s[:3]+[16]) # reshape upconvolution to have proper shape
    d4 = tf.concat([d2, d4], 3) # <- unet like concat first with last
    _, d4 = dense(d4, 16, pool=False)
    print('d4', d4)

    d5 = tf.depth_to_space(d4, 2)
    d5 = tf.reshape(d5, shape=d1s[:3]+[8])
    d5 = tf.concat([d1, d5], 3)
    _, d5 = dense(d5, 8, pool=False)
    print('d5', d5)

    d5 = GLU(d5, 32, [64, 1], scope='glu32')
    d5 = GLU(d5, 128, [64, 1], scope='glu128')

    conv5 = slim.conv2d(d5, discrete_class, [1,1], scope='conv5', activation_fn=lrelu)
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, d1s[:2]+[discrete_class])
    print('out: ', out)
    return out

