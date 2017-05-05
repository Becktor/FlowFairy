import tensorflow as tf

def embedding(l, cls, embedding_size, num_classes):

    if embedding_size > 1:
        shape = [num_classes]
    else:
        shape = [num_classes, embedding_size]

    W = tf.get_variable('embedding', shape, initializer=tf.truncated_normal_initializer())
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
