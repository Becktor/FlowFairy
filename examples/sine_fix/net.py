import tensorflow as tf
from flowfairy.conf import settings

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE

# Create some wrappers for simplicity
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
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


# Create model
def conv_net(x, weights, biases, dropout):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    #convblock 1
    conv1 = conv2d(xs, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=4)
    #print('conv1: ', pool1)

    ##convblock 4
    #conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    #pool2 = maxpool2d(conv2, k=2)
    #print('conv2: ', pool2)

    ##convblock 3
    #conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
    #print('conv3: ', conv3)

    ##convblock 4
    conv4 = tf.depth_to_space(conv1, 2) #upconv
    conv4 = tf.reshape(conv4, shape=[32, sr, 1, 1]) # reshape upconvolution to have proper shape
    #conv4 = conv2d(conv4, weights['wc4'], biases['bc4'])
    #print('conv4: ', conv4)

    ##convblock 5
    #conv5 = tf.concat([conv1, conv4], 3) # <- unet like concat first with last
    #conv5 = conv2d(conv5, weights['wc5'], biases['bc5'])
    #print('conv5: ', conv5)

    out = tf.reshape(conv4, [-1, weights['out'].get_shape().as_list()[0]])
    #print(out)
    #out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    return out #, conv1, conv2, conv3

class Net:

    def init(self, x, y, m, keep_prob):
        # Store layers weight & bias

        weights = {
            'wc1': tf.Variable(tf.random_normal([64, 1, 1, 4])),
            'wc2': tf.Variable(tf.random_normal([64, 1, 4, 8])),
            'wc3': tf.Variable(tf.random_normal([64, 1, 8, 4])),
            'wc4': tf.Variable(tf.random_normal([64, 1, 1, 4])),
            'wc5': tf.Variable(tf.random_normal([64, 1, 8, 1])),
            'out': tf.Variable(tf.random_normal([sr, 1]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([4])),
            'bc2': tf.Variable(tf.random_normal([8])),
            'bc3': tf.Variable(tf.random_normal([4])),
            'bc4': tf.Variable(tf.random_normal([4])),
            'bc5': tf.Variable(tf.random_normal([1])),
            'out': tf.Variable(tf.random_normal([sr]))
        }

        self.x = x
        self.y = y

        # Construct model
        pred = conv_net(x, weights, biases, keep_prob)
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
