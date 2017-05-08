import tensorflow as tf
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
embedding_size = settings.EMBEDDING_SIZE
embedding_input_size = settings.EMBEDDING_INPUT_SIZE
num_classes = settings.CLASS_COUNT

def expand(l, emb, embedding_size):
    shape = l.get_shape().as_list()[:-1]
    zeros = tf.zeros(shape + [embedding_size])

    #emb = tf.expand_dims(emb, 1)
    emb = tf.expand_dims(emb, 1)
    emb = emb[:, None, :, :] + zeros
    return emb


# Create model
def conv_net(x, cls, weights, biases, dropout):
    xs = tf.reshape(x, shape = [-1, sr, 1, 1] )
    wc1, wc2, wc3, wc4, wc5, wout = weights

    #convblock 1
    with tf.variable_scope(wc1['name']):
        conv1 = conv2d(xs, wc1['shape'])
        pool1 = maxpool2d(conv1, k=2)
    print('conv1: ', pool1)

    cls = cls[-1, :embedding_input_size]
    cls = tf.reshape(cls, [-1, embedding_input_size, 1, 1])

    with tf.name_scope('embedding'):
        with tf.variable_scope('embed1'):
            embed = conv2d(cls, [64, 1, 1, 4], strides=8)
            embed = maxpool2d(embed, k=4)

        with tf.variable_scope('embed2'):
            embed = conv2d(embed, [8, 1, 4, num_classes], strides=4)
            embed = avgpool2d(embed, k=16)
        embed = tf.reshape(embed, [-1, num_classes])
        x_cls = tf.nn.softmax(embed)
        x_cls = tf.argmax(x_cls, axis=1)
        print(x_cls.shape)

        with tf.variable_scope('embedding'):
            emb1 = embedding(pool1, x_cls, embedding_size, num_classes)
        expanded = expand(pool1, emb1, embedding_size)
        expanded = tf.concat([pool1, expanded], axis=3)

    #convblock 2
    with tf.variable_scope(wc2['name']):
        conv2 = conv2d(expanded, wc2['shape'])
        pool2 = maxpool2d(conv2, k=2)
    print('conv2: ', pool2)

    #convblock 3
    with tf.variable_scope(wc3['name']):
        conv3 = conv2d(pool2, wc3['shape'])
    print('conv3: ', conv3)

    #convblock 4
    conv4 = tf.depth_to_space(conv3, 2) #upconv
    c1shape = conv1.get_shape().as_list()
    conv4 = tf.reshape(conv4, shape=[c1shape[0], sr, 1, -1]) # reshape upconvolution to have proper shape

    with tf.variable_scope(wc4['name']):
        conv4 = conv2d(conv4, wc4['shape'])
    print('conv4: ', conv4)

    #convblock 5
    conv5 = tf.concat([conv4, conv1], 3) # <- unet like concat first with last

    with tf.variable_scope(wc5['name']):
        conv5 = conv2d(conv5, wc5['shape'])
    print('conv5: ', conv5)

    #out
    out = tf.reshape(conv5, [-1, wout['shape'][0], 256])
    print('out: ', out)
    return out, embed

class Net:

    def __init__(self):
        # Store layers weight & bias

        weights = [
            ('wc1', [128, 1, 1, 4]),
            ('wc2', [64, 1, embedding_size+4, 18]),
            ('wc3', [32, 1, 18, 4]),
            ('wc4', [32, 1, 1, 8]),
            ('wc5', [8, 1, 12, 256]),
            ('out', [sr, 256])
        ]
        self.weights = [ {'name': name, 'shape': shape} for name, shape in weights ]

        biases = [
            ('bc1', [4]),
            ('bc2', [18]),
            ('bc3', [4]),
            ('bc4', [8]),
            ('bc5', [256]),
            ('out', [sr])
        ]

    def feedforward(self, x, y, frqid, keep_prob, cls):
        pred, embedding = conv_net(x, cls, self.weights, None, keep_prob)

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

        return pred, embedding, cost, accuracy

    def train(self, **kwargs):
        self.train_x = kwargs['x']
        self.train_y = kwargs['y']

        self.train_pred, self.train_embedding, self.train_cost, self.train_acc = self.feedforward(**kwargs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.train_cost)


    def validation(self, **kwargs):
        self.val_x = kwargs['x']
        self.val_y = kwargs['y']

        self.val_pred, self.val_embedding, self.val_cost, self.val_acc = self.feedforward(**kwargs)

    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
