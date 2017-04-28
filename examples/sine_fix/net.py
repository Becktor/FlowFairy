import tensorflow as tf
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
embedding_size = settings.EMBEDDING_SIZE
num_classes = settings.CLASS_COUNT


# Create model
def conv_net(x, cls, weights, biases, dropout):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = conv2d(xs, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d(conv1, k=2)
    print('conv1: ', pool1)

    emb1 = embedding(pool1, cls, embedding_size, num_classes)

    #convblock 2
    conv2 = conv2d(emb1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d(conv2, k=2)
    print('conv2: ', pool2)

    #convblock 3
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 2) #upconv
    #conv4 = tf.reshape(conv4, shape=[-1, sr, 1, 1]) # reshape upconvolution to have proper shape
    out = tf.reshape(conv4, [-1, weights['out'].get_shape().as_list()[0]])
    #print(out)
    #out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    return out #, conv1, conv2, conv3

class Net:

    def init(self, x, y, keep_prob, frqid, m):
        # Store layers weight & bias

        weights = {
            'wc1': tf.Variable(tf.truncated_normal([256, 1, 1, 4])),
            'wc2': tf.Variable(tf.truncated_normal([128, 1, embedding_size+4, embedding_size//4])),
            'wc3': tf.Variable(tf.truncated_normal([64, 1, embedding_size//4, 4])),
            'wc4': tf.Variable(tf.truncated_normal([64, 1, 1, 8])),
            'wc5': tf.Variable(tf.truncated_normal([1, 1, 8, 256])),
            'out': tf.Variable(tf.truncated_normal([sr, 256]))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([4])),
            'bc2': tf.Variable(tf.truncated_normal([embedding_size//4])),
            'bc3': tf.Variable(tf.truncated_normal([4])),
            'bc4': tf.Variable(tf.truncated_normal([8])),
            'bc5': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([sr]))
        }

        self.x = x
        self.y = y

        # Construct model
        pred = conv_net(x, frqid, weights, biases, keep_prob)
        self.pred = pred

        # Define loss and optimizer
        with tf.name_scope('cost'):
             self.cost = tf.sqrt(tf.reduce_mean(tf.square(y*m - pred*m)))

        with tf.name_scope('l1'):
             self.l1 = tf.reduce_mean(tf.abs(pred*m - y*m))

        with tf.name_scope('l2'):
             self.l2 = tf.reduce_mean(tf.pow(pred*m - y*m, 2))

        self.loss = self.l1

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.l1)

        # Evaluate model
        #accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))

    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
