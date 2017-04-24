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
    xs = tf.reshape(x, shape=[-1, sr, 1, 1])
    # Convolution Layer
    conv1 = conv2d(xs, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    print('conv1: ', conv1)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    print('conv2: ', conv2)
    #conv3 = SubpixelConv2d(x, scale=2, n_out_channel=2, name='subpixel_conv2d2')
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    print('conv3: ', conv3)
    #conv4 = PS(conv3, 2)
    conv4 = tf.depth_to_space(conv3, 2)
    print('conv4: ', conv4)
    #out=
    out = tf.reshape(conv3, [-1, weights['out'].get_shape().as_list()[0]])
    print(out)
    #out = tf.add(tf.matmul(out, weights['out']), biases['out'])
    #print(out)
    return out , conv1, conv2, conv3


class Net:

    its = 0

    def init(self, x, y, m, keep_prob):
         # Store layers weight & bias
        weights = {

            'wc1': tf.Variable(tf.random_normal([128, 1, 1, 8])),

            'wc2': tf.Variable(tf.random_normal([256, 1, 8, 8])),

            'wc3': tf.Variable(tf.random_normal([128, 1, 8, 4])),

            'out': tf.Variable(tf.random_normal([samplerate, 1]))
        }

        biases = {
             'bc1': tf.Variable(tf.random_normal([8])),
             'bc2': tf.Variable(tf.random_normal([8])),
             'bc3': tf.Variable(tf.random_normal([4])),
             'out': tf.Variable(tf.random_normal([samplerate]))
        }

        # Construct model
        pred, c1, c2, c3 = conv_net(x, weights, biases, keep_prob)

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
        accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, pred))))
        # Initializing the variables
        self.init = tf.global_variables_initializer()

    def begin(self, session):
         session.run(self.init)

    def train(self, sess):
         self.its += 1
         sess.run(self.optimizer)

    def should_stop(self):
        return False

    def display(self, sess, step):
         it = step * batch_size
         l1, l2, loss = sess.run([self.l1, self.l2, self.cost])
         print(f'Iter {it}, l1: {l1:.6f}, l2: {l2:.6f}, loss: {loss:.6f}')
