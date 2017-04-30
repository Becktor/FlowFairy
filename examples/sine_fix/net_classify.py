import tensorflow as tf
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
embedding_size = settings.EMBEDDING_SIZE
num_classes = settings.CLASS_COUNT

def expand(l, emb, embedding_size):
    shape = l.get_shape().as_list()[:-1]
    zeros = tf.zeros(shape + [embedding_size])

    emb = emb[:, None, :, :] + zeros
    return emb


# Create model
def conv_net(x, cls, weights, biases, dropout):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = conv2d(xs, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d(conv1, k=2)
    print('conv1: ', pool1)

    emb1 = embedding(pool1, cls, embedding_size, num_classes)
    expanded = expand(pool1, emb1, embedding_size)
    expanded = tf.concat([pool1, expanded], axis=3)

    #convblock 2
    conv2 = conv2d(expanded, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d(conv2, k=2)
    print('conv2: ', pool2)

    #convblock 3
    conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 2) #upconv
    c1shape = conv1.get_shape().as_list()
    conv4 = tf.reshape(conv4, shape=[c1shape[0], sr, 1, -1]) # reshape upconvolution to have proper shape
    conv4 = conv2d(conv4, weights['wc4'], biases['bc4'])
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, weights['out'].get_shape().as_list()[0], 256])
    return out, emb1

class Net:

    def init(self, x, y, keep_prob, frqid):
        # Store layers weight & bias

        weights = {
            'wc1': tf.Variable(tf.truncated_normal([128, 1, 1, 8])),
            'wc2': tf.Variable(tf.truncated_normal([64, 1, embedding_size+8, embedding_size+16])),
            'wc3': tf.Variable(tf.truncated_normal([64, 1, embedding_size+16, embedding_size+32])),
            'wc4': tf.Variable(tf.truncated_normal([128, 1, 24, 8])),
            'wc5': tf.Variable(tf.truncated_normal([32, 1, embedding_size//8, 256])),
            'out': tf.Variable(tf.truncated_normal([sr, 256]))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([8])),
            'bc2': tf.Variable(tf.truncated_normal([embedding_size+16])),
            'bc3': tf.Variable(tf.truncated_normal([embedding_size+32])),
            'bc4': tf.Variable(tf.truncated_normal([8])),
            'bc5': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([sr]))
        }

        self.x = x
        self.y = tf.cast(y, tf.int64)

        # Construct model
        pred, self.embedding = conv_net(x, frqid, weights, biases, keep_prob)
        self.pred = pred

        target_output = tf.reshape(self.y,[-1])
        prediction = tf.reshape(pred,[-1, discrete_class])

        # Define loss and optimizer
        with tf.name_scope('cost'):
            self.sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction,
                                                                    labels = target_output)
            self.cost = tf.reduce_mean(self.sparse)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 2), self.y)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
