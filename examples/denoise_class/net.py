import tensorflow as tf
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
discrete_class = settings.DISCRETE_CLASS

def conv_net(x, weights, biases, dropout):
    xs = tf.reshape(x, shape = [32, sr, 1, 1] )
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
    conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 2]) # reshape upconvolution to have proper shape
    conv4 = conv2d(conv4, weights['wc4'], biases['bc4'])
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    conv5 = conv2d(conv5, weights['wc5'], biases['bc5'])
    print('conv5: ', conv5)
    #out
    out = tf.reshape(conv5, [-1, weights['out'].get_shape().as_list()[0], 256])
    print('out: ', out)
    return out

class Net:

    def __init__(self):
        # Store layers weight & bias

        self.weights = {
            'wc1': tf.Variable(tf.truncated_normal([256, 1, 1, 4])),
            'wc2': tf.Variable(tf.truncated_normal([128, 1, 4, 16])),
            'wc3': tf.Variable(tf.truncated_normal([128, 1, 16, 8])),
            'wc4': tf.Variable(tf.truncated_normal([128, 1, 2, 16])),
            'wc5': tf.Variable(tf.truncated_normal([1, 1, 20, 256])),
            'out': tf.Variable(tf.truncated_normal([sr, 256]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.truncated_normal([4])),
            'bc2': tf.Variable(tf.truncated_normal([16])),
            'bc3': tf.Variable(tf.truncated_normal([8])),
            'bc4': tf.Variable(tf.truncated_normal([16])),
            'bc5': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([sr]))
        }


    def feedforward(self, x, y, chunk, keep_prob):
        pred = conv_net(x, self.weights, self.biases, dropout)

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
