import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model1 as model
from flowfairy.conf import settings
from util import lrelu, conv2d, maxpool2d, GLU, causal_GLU

batch_size = settings.BATCH_SIZE
samplerate = sr = settings.SAMPLERATE
dropout = settings.DROPOUT
learning_rate = settings.LEARNING_RATE
discrete_class = settings.DISCRETE_CLASS


class Net:

    def __init__(self):
        pass

    def feedforward(self, x, y, chunk, is_training=False):
        pred = model.conv_net(x, is_training)

        target_output = tf.reshape(y,[-1])
        prediction = tf.reshape(pred,[-1, discrete_class])

        # Define loss and optimizer
        with tf.name_scope('cost'):
            cost = tf.losses.sparse_softmax_cross_entropy(logits = prediction,
                                                          labels = target_output,
                                                          scope='xentropy')

        correct_pred = tf.equal(tf.argmax(pred, 2), y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return pred, cost, accuracy, chunk

    def train(self, **kwargs):
        with tf.name_scope('train'):
            self.train_x = kwargs['x']
            self.train_y = kwargs['y']

            self.train_pred, self.train_cost, self.train_acc, self.train_chunk = self.feedforward(is_training=True, **kwargs)
            self.optimizer = ops.train()

    def validation(self, **kwargs):
        with tf.name_scope('val'):
            self.val_x = kwargs['x']
            self.val_y = kwargs['y']

            self.val_pred, self.val_cost, self.val_acc, self.val_chunk = self.feedforward(**kwargs)



    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
