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
    conv1 = slim.conv2d(xs, 4, [256,1], activation_fn=lrelu, scope='conv1')
    print('conv1: ', conv1)
    pool1 = slim.max_pool2d(conv1, [2, 1], scope='pool1')
    print('pool1: ', pool1)

    #convblock 2
    conv2 = slim.conv2d(pool1, 16, [128, 1],
                        normalizer_fn=slim.batch_norm, activation_fn=lrelu, scope='conv2')
    print('conv2: ', conv2)
    pool2 = slim.max_pool2d(conv2, [2, 1], scope='pool2')
    print('pool2: ', pool2)

    #convblock 3
    conv3 = slim.conv2d(pool2, 16, [128, 1], activation_fn=lrelu, scope='conv3')
    print('conv3: ', conv3)

    #convblock 4
    d2s4 = tf.depth_to_space(conv3, 4) #upconv
    print('depth2space: ', d2s4)
    # reshape upconvolution to have proper shape
    conv4 = tf.reshape(d2s4, shape=[-1, sr, 1, 4])
    conv4 = slim.conv2d(conv4, 16, [128, 1], activation_fn=lrelu, scope='conv4')
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet concat conv1 with conv4 16+4
    print('concat: ', conv5)
    conv5 = slim.conv2d(conv5, 256, [1, 1], activation_fn=lrelu, scope='conv5')
    print('conv5: ', conv5)
    #out
    out = tf.reshape(conv5, [-1, sr, 256])
    print('out: ', out)
    return out

class Net:

    def __init__(self):
        # Store layers weight & bias
        pass

    def feedforward(self, x, y, chunk, keep_prob):
        pred = conv_net(x, dropout)

        target_output = tf.reshape(y,[-1])
        prediction = tf.reshape(pred,[-1, discrete_class])

        # Define loss and optimizer
        with tf.name_scope('cost'):
            sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction,
                                                                    labels = target_output)
            cost = tf.reduce_mean(sparse)

        correct_pred = tf.equal(tf.argmax(pred, 2), y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
