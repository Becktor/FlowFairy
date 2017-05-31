import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers

def embedding(cls, embedding_size, num_classes, scope=None):

    shape = [num_classes, embedding_size]

    W = tf.get_variable('embedding', shape, initializer=initializers.xavier_initializer())
    emb = tf.nn.embedding_lookup(W, cls)

    return emb

# Create some wrappers for simplicity
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.name_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(x, kernel_shape, strides=1):
    # Conv2D wrapper, with bias and relu activation
    W = tf.get_variable('weights', kernel_shape, initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('bias', [kernel_shape[-1]], initializer=tf.truncated_normal_initializer())
    x = tf.nn.conv2d(x, W, strides=[1, strides, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return lrelu(x)

def upconv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d_transpose(x, W, output_shape=(batch_size, sr, 1, 1),
                               strides=(1, strides, strides, 1), padding='SAME')
    x = tf.nn.bias_add(x, b)
    return lrelu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def GLU(x, num_filters, kernel_size, scope='glu', **kwargs):
    with tf.name_scope('glu_'+scope):
        A = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_unit', activation_fn=None, **kwargs)
        B = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_gate', activation_fn=None, **kwargs)
        return A * tf.sigmoid(B)

def reluGLU(x, num_filters, kernel_size, scope='glu', **kwargs):
    with tf.name_scope('rglu_'+scope):
        A = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_unit', activation_fn=None, **kwargs)
        B = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_gate', activation_fn=None, **kwargs)
        return A * tf.nn.relu6(B)

def GTU(x, num_filters, kernel_size, scope='gtu', **kwargs):
    A = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_unit', activation_fn=None, **kwargs)
    B = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_gate', activation_fn=None, **kwargs)
    return tf.tanh(A) * tf.sigmoid(B)

def causal_conv(conv):
    def causal(x, num_filters, kernel_size, scope='causal', **kwargs):
        with tf.name_scope('causal_'+scope):
            width = kernel_size[0]//2
            pad_size = x.get_shape().as_list()
            pad_size[1] = width
            x = tf.concat([tf.zeros(pad_size), x], axis=1)[:,:-width]
            return conv(x, num_filters, kernel_size, scope, **kwargs)
    return causal

causal_GLU = causal_conv(GLU)

DEG_TO_RAD = np.pi / 180
SHIFT_IN_RAD = np.pi / 2.0 # shift zero to be in front
SPEECH_INTERVAL_RAD = 15*DEG_TO_RAD
MICDIST = 0.1
SPEED_OF_SOUND = 340 # m/s

def circle_coord(rad, r=1):
    x = r*tf.cos(rad)
    y = r*tf.sin(rad)
    return tf.concat([x, y], 1)

def random_noisy_speech(speech, noise, outputlen, micdist, sr=22050):
    # sample single random pos
    bs, seq_len, _ = speech.get_shape().as_list()
    pos = tf.random_uniform([1, 1], 0, np.float32(np.pi))
    cord = circle_coord(pos, 1.0)

    # dist to each microphone vector of shape bs
    dst_mic_L = tf.sqrt((-micdist - cord[:, 0]) ** 2 + (0 - cord[:, 1]) ** 2)
    dst_mic_R = tf.sqrt((micdist - cord[:, 0]) ** 2 + (0 - cord[:, 1]) ** 2)

    # number of sample offset to each microphone.
    offset_L = tf.cast(tf.round((dst_mic_L / SPEED_OF_SOUND) * sr), tf.int32)
    offset_R = tf.cast(tf.round((dst_mic_R / SPEED_OF_SOUND) * sr), tf.int32)
    # set minimum offset to 0
    min_offset = tf.reduce_min(tf.concat([offset_L, offset_R], axis=0))
    offset_L -= min_offset
    offset_R -= min_offset

    noise_L = noise[:, offset_L[0]:]
    noise_R = noise[:, offset_R[0]:]
    noise_L = noise_L[:, :outputlen]
    noise_R = noise_R[:, :outputlen]
    speech = speech[:, :outputlen]

    x_left = speech + noise_L
    x_right = speech + noise_R
    x = tf.concat([x_left, x_right], axis=2)
    y = tf.concat([speech, speech], axis=2)

    # dummies
    speech_rad = tf.zeros((bs, 1))
    noise_rad = tf.zeros_like(speech_rad)

    noise_rad += pos[0] - (np.pi/2.0)

    # shift noise_Rad to lie in -pi/2, pi/2 instead of 0,pi

    return x, y, speech_rad, noise_rad
