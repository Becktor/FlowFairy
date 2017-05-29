import ops
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model1 as model
from flowfairy.conf import settings

discrete_class = settings.DISCRETE_CLASS

class Net:

    def __init__(self):
        pass

    def feedforward(self, x, y, frqid, frqid2, is_training=False):
        pred = model.conv_net(x, frqid, None, is_training)

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

        return pred, cost, accuracy

    def train(self, **kwargs):
        with tf.name_scope('train'):
            self.train_x = kwargs['x']
            self.train_y = kwargs['y']

            self.train_pred, self.train_cost, self.train_acc = self.feedforward(is_training=True, **kwargs)
            self.optimizer = ops.train()

    def validation(self, **kwargs):
        with tf.name_scope('val'):
            self.val_x = kwargs['x']
            self.val_y = kwargs['y']

            self.val_pred, self.val_cost, self.val_acc = self.feedforward(**kwargs)
            self.val_pred = tf.Print(self.val_pred,
                                     [kwargs['frqid'],
                                      kwargs['frqid2']],
                                     message='frqids: ')

    def begin(self, session):
        #session.run(self.init)
        pass

    def should_stop(self):
        return False
