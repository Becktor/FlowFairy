import tensorflow as tf
import tensorflow.contrib.slim as slim

from flowfairy.conf import settings

learning_rate = settings.LEARNING_RATE

def train():
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    total_loss = tf.losses.get_total_loss()

    return slim.learning.create_train_op(
        total_loss,
        optimizer,
        clip_gradient_norm=5
    )
