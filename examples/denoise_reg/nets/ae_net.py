import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
discrete_class = settings.DISCRETE_CLASS

def conv_net(x,  dropout):
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

class Net:

    def __init__(self):
        # Store layers weight & bias
        pass

    def feedforward(self, x, y, chunk, keep_prob):
        pred = conv_net(x, dropout)

        with tf.name_scope('cost'):
            cost = tf.sqrt(tf.reduce_mean(tf.square(y - pred)))

        with tf.name_scope('l1'):
            l1 = tf.reduce_mean(tf.abs(pred - y))

        with tf.name_scope('l2'):
            l2 = tf.reduce_mean(tf.pow(pred - y, 2))
        # Evaluate model

        with tf.name_scope('accuracy'):
            accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))

        return pred, cost, accuracy, chunk

    def train(self, **kwargs):
        self.train_x = kwargs['x']
        self.train_y = kwargs['y']

        self.train_pred, self.train_cost, self.train_acc, self.train_chunk = self.feedforward(**kwargs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.train_cost)


    def validation(self, **kwargs):
        self.val_x = kwargs['x']
        self.val_y = kwargs['y']

        self.val_pred, self.val_cost, self.val_acc, self.val_chunk = self.feedforward(**kwargs)


    def begin(self, session):
        pass

    def should_stop(self):
        return False
