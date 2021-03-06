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


# Create model
def conv_net(x, cls, dropout, is_training=False):
    xs = tf.expand_dims(x, -1)
    xs = tf.expand_dims(xs, -1)

    conv1 = causal_GLU(xs, 4, [128, 1], scope='conv1_1', normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training, 'decay': 0.9})
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



    #conv1 = GLU(conv1, 4, [256, 1], scope='conv1_2')

    #with tf.name_scope('embedding'):

    #convblock 2
    conv2 = GLU(conv1, 8, [128, 1], scope='conv2_1')
    conv2 = GLU(conv2, 8, [128, 1], scope='conv2_2')
    conv2 = slim.max_pool2d(conv2, [2,1])
    print('conv2: ', conv2)

    with tf.variable_scope('embedding'):
        emb1 = embedding(cls, embedding_size, num_classes)
        embedded = broadcast(conv2, emb1)
    print('embedded:', embedded)

    #convblock 3
    conv3 = GLU(embedded, 16, [128, 1], scope='conv3_1')
    conv3 = GLU(conv3, 16, [128, 1], scope='conv3_2')
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 4) #upconv
    print('d2sp: ', conv4)
    conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 8]) # reshape upconvolution to have proper shape

    conv4 = GLU(conv4, 16, [128, 1], scope='conv4_1')
    #convblock 5
    conv4 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    conv4 = GLU(conv4, 16, [128, 1], scope='conv4_2')
    print('conv4: ', conv4)


    conv5 = GLU(conv4, discrete_class, [2,1], scope='conv5')
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, sr, discrete_class])
    print('out: ', out)
    return out

