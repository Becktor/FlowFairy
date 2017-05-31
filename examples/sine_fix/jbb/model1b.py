import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
embedding_size = settings.EMBEDDING_SIZE
num_classes = settings.CLASS_COUNT

def broadcast(l, emb):
    sh = l.get_shape().as_list()[1]
    emb = emb[:, None, None, :]
    emb = tf.tile(emb, (1,sh,1,1))
    return tf.concat([l, emb], 3)

# Create model
def conv_net(x, cls, dropout, is_training=False):
    with tf.name_scope('input'):
        xs = tf.expand_dims(x, -2)

    conv1 = GLU(xs, 4, [256, 1],  scope='conv1_1',
                normalizer_fn=slim.batch_norm,
                normalizer_params={'is_training': is_training,
                                   'decay': 0.9})
    print('conv1: ', conv1)
    conv1 = GLU(conv1, 4, [256, 1], scope='conv1_2')
    #pool1 = slim.max_pool2d(conv1, [2,1])
    print('conv1: ', conv1)


    #convblock 2
    conv2 = GLU(conv1, 8, [128, 1], scope='conv2_1')
    print('conv2: ', conv2)
    conv2 = GLU(conv2, 8, [128, 1], scope='conv2_2')
    print('conv2: ', conv2)
    pool2 = slim.max_pool2d(conv2, [2,1])
    print('conv2: ', pool2)

    #convblock 3
    with tf.variable_scope('embedding'):
        emb1 = embedding(cls, embedding_size, num_classes)
        embedded = broadcast(pool2, emb1)
        print('embedded:', embedded)

    conv3 = GLU(embedded, 16, [128, 1], scope='conv3_1')
    conv3 = GLU(conv3, 16, [128, 1], scope='conv3_2')
    print('conv3: ', conv3)

    with tf.name_scope('d2s1'):
        conv4 = tf.depth_to_space(conv3, 4) #upconv
        conv1shape = conv1.get_shape().as_list()
        conv4 = tf.reshape(conv4, shape=conv1shape[:3]+[8]) # reshape upconvolution to have proper shape
        print('d2sp: ', conv4)

    conv4 = GLU(conv4, 16, [128, 1], scope='conv4_1')
    with tf.name_scope('concat'):
        conv4 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    conv4 = GLU(conv4, 32, [128, 1], scope='conv4_2')
    conv4 = GLU(conv4, 64, [128, 1], scope='conv4_3')
    print('conv4: ', conv4)

    conv5 = slim.conv2d(conv4, discrete_class, [1,1], scope='conv5')
    print('conv5: ', conv5)

    with tf.name_scope('output'):
        out = tf.reshape(conv5, [-1, conv1shape[1], discrete_class])
    print('out: ', out)
    return out
