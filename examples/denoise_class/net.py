import tensorflow as tf
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
discrete_class = settings.DISCRETE_CLASS

def conv_net(x, weights, biases, dropout):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = conv2d(xs, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d(conv1, k=2)
    print('conv1: ', pool1)

    #convblock 2
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d(conv2, k=2)
    print('conv2: ', pool2)

    #convblock 3
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 2) #upconv
    conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 1]) # reshape upconvolution to have proper shape
    conv4 = conv2d(conv4, weights['wc4'], biases['bc4'])
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, weights['out'].get_shape().as_list()[0], 256])
    print('out: ', out)
    return out

class Net:

    def init(self, x, y, m, keep_prob):
        # Store layers weight & bias

        weights = {
            'wc1': tf.Variable(tf.truncated_normal([64, 1, 1, 4])),
            'wc2': tf.Variable(tf.truncated_normal([64, 1, 4, 8])),
            'wc3': tf.Variable(tf.truncated_normal([64, 1, 8, 4])),
            'wc4': tf.Variable(tf.truncated_normal([64, 1, 1, 8])),
            'wc5': tf.Variable(tf.truncated_normal([1, 1, 8, 256])),
            'out': tf.Variable(tf.truncated_normal([sr, 256]))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([4])),
            'bc2': tf.Variable(tf.truncated_normal([8])),
            'bc3': tf.Variable(tf.truncated_normal([4])),
            'bc4': tf.Variable(tf.truncated_normal([8])),
            'bc5': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([sr]))
        }

        self.x = x
        self.y = tf.cast(y, tf.int64)

        # Construct model
        pred = conv_net(self.x, weights, biases, keep_prob)
        self.pred = pred

        # Define loss and optimizer
        # Construct model and define variables
        pred = conv_net(self.x, weights, biases, keep_prob)

        target_output = tf.reshape(self.y,[-1])
        prediction = tf.reshape(pred,[-1, discrete_class])

        # Define loss and optimizer
        with tf.name_scope('cost'):
            self.sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction,
                                                                    labels = target_output)
            self.cost = tf.reduce_mean(self.sparse)

        #Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 2), self.y)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.loss = self.cost


    def begin(self, session):
        pass

    def should_stop(self):
        return False
