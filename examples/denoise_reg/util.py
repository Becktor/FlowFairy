import tensorflow as tf
import tensorflow.contrib.slim as slim

def embedding(cls, embedding_size, num_classes):

    shape = [num_classes, embedding_size]

    W = tf.get_variable('embedding', shape, initializer=tf.truncated_normal_initializer())
    emb = tf.nn.embedding_lookup(W, cls)

    return emb

def diconv2d(x, num_filters, rate, scope='conv', **kwargs):
    with tf.variable_scope('diluted_'+scope) as scope:
        c = slim.batch_norm(x)
        W = tf.get_variable('weights', num_filters, initializer=tf.truncated_normal_initializer())
        c = tf.nn.atrous_conv2d(c, W, rate, padding='SAME')
        c = tf.nn.relu(c)
    return c

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

def dconv(x, num_filters, kernel_size, activation_fn=lrelu, scope='dense', **kwargs):
    with tf.variable_scope(scope) as scope:
        c = slim.batch_norm(x)
        c = tf.nn.relu(c)
        c = slim.conv2d(x, num_filters, kernel_size, activation_fn=None)
    return tf.concat([x,c], 3)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def avgpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def GLU(x, num_filters, kernel_size, scope='glu', **kwargs):
    A = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_unit', activation_fn=None, **kwargs)
    B = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_gate', activation_fn=None, **kwargs)
    return A * tf.sigmoid(B)

def GTU(x, num_filters, kernel_size, scope='gtu', **kwargs):
    A = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_unit', activation_fn=None, **kwargs)
    B = slim.conv2d(x, num_filters, kernel_size, scope=scope+'_gate', activation_fn=None, **kwargs)
    return tf.tanh(A) * tf.sigmoid(B)

def causal_conv(conv):
    def causal(x, num_filters, kernel_size, scope='causal', **kwargs):
        width = kernel_size[0]//2
        pad_size = x.get_shape().as_list()
        pad_size[1] = width
        x = tf.concat([tf.zeros(pad_size), x], axis=1)[:,:-width]
        return conv(x, num_filters, kernel_size, scope, **kwargs)
    return causal

causal_GLU = causal_conv(GLU)
