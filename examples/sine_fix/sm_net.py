import tensorflow as tf
import tensorflow.contrib.slim as slim
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, embedding, avgpool2d, GLU, causal_GLU

discrete_class = settings.DISCRETE_CLASS
batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
embedding_size = settings.EMBEDDING_SIZE
num_classes = settings.CLASS_COUNT

def broadcast(l, emb):
    sh = l.get_shape().as_list()[1]
    emb = emb[:, None, None, :]
    emb = tf.tile(emb, (1,sh,1,1))
    return tf.concat([l, emb], 3)


# Create model
def conv_net(x, cls, dropout, is_training=False):
    xs = tf.reshape(x, shape=[-1, 11024, 1, 1])

    conv1 = GLU(xs, 4, [128, 1], normalizer_fn=slim.batch_norm, scope='conv1_1')
    #conv1 = GLU(conv1, 8, [128, 1], scope='conv1_2')
    pool1 = slim.max_pool2d(conv1, [2,1], scope='pool1')
    print('conv1: ', conv1)

    conv2 = GLU(pool1, 8, [128, 1], scope='conv2_1')
    #conv2 = GLU(conv2, 16, [128, 1], scope='conv2_2')
    pool2 = slim.max_pool2d(conv2, [2,1], scope='pool2')
    print('conv2: ', conv2)

    conv3 = GLU(pool2, 64, [128, 1], scope='conv3_1')
    #conv3 = GLU(conv3, 64, [128, 1], scope='conv3_2')
    pool3 = slim.max_pool2d(conv3, [2,1], scope='pool3')
    print('conv3: ', conv3)

    conv4 = GLU(pool3, 64, [128, 1], scope='conv4_1')
    #conv4 = GLU(conv4, 64, [128, 1], scope='conv4_2')
    pool4 = slim.max_pool2d(conv4, [2,1], scope='pool4')
    print('conv4: ', conv4)

    with tf.variable_scope('embedding'):
        emb1 = embedding(cls, embedding_size, num_classes)
        embedded = broadcast(pool4, emb1)
        print('embedded:', embedded)

    conv5 = GLU(embedded, 64, [128, 1], scope='conv5_1')
    #conv5 = GLU(conv5, 64, [128, 1], scope='conv5_2')
    print('conv5: ', conv5)

    with tf.name_scope('d2s1'):
        conv6 = tf.depth_to_space(conv5, 4);print('d2s_6: ', conv6)
        conv6 = tf.reshape(conv6, shape=[-1, 2756, 1, 16])
    with tf.name_scope('concat1'):
        conv6 = tf.concat([conv6, conv3], axis=3)
    #conv6 = GLU(conv6, 64, [128, 1], scope='conv6_1')
    conv6 = GLU(conv6, 64, [128, 1], scope='conv6_2')
    print('conv6: ', conv6)

    with tf.name_scope('d2s2'):
        conv7 = tf.depth_to_space(conv6, 4);print('d2s_7: ', conv7)
        conv7 = tf.reshape(conv7, shape=[-1, 11024, 1, 16])
    with tf.name_scope('concat2'):
            conv7 = tf.concat([conv7, conv1], axis=3)
   # conv7 = GLU(conv7, 64, [128, 1], scope='conv8_1')
    conv7 = GLU(conv7, 64, [128, 1], scope='conv8_2')
    print('conv7: ', conv7)

    clsifyer = GLU(conv7, discrete_class, [1,1], scope='clsifyer')
    print('clsifyer: ', clsifyer)

    #out
    with tf.name_scope('output'):
        out = tf.reshape(clsifyer, [-1, sr, 256])
    print('out: ', out)
    return out

class Net:

    def __init__(self):
        pass

    def feedforward(self, x, y, frqid, is_training=False):
        pred = conv_net(x, frqid, None, is_training)

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

        return pred, cost, accuracy

    def train(self, **kwargs):
        with tf.name_scope('train'):
            self.train_x = kwargs['x']
            self.train_y = kwargs['y']

            self.train_pred, self.train_cost, self.train_acc = self.feedforward(is_training=True, **kwargs)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.train_cost)
        #gradients, variables = zip(*optimizer.compute_gradients(self.train_cost))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #self.optimizer = optimizer.apply_gradients(zip(gradients, variables))

    def validation(self, **kwargs):
        with tf.name_scope('val'):
            self.val_x = kwargs['x']
            self.val_y = kwargs['y']

            self.val_pred, self.val_cost, self.val_acc = self.feedforward(**kwargs)

    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
