import tensorflow as tf
import numpy as np
import itertools as it
import importlib

from flowfairy.conf import settings
from flowfairy.utils import take
from flowfairy.data.loader import DataLoader
from flowfairy.feature import FeatureManager
from flowfairy.core.queue import FlowQueue


def load_net():
    net = importlib.import_module(settings.NET).Net()
    return net

def run(*args, **options):
    with tf.Graph().as_default():
        loader = DataLoader()

        fts = FeatureManager(loader)

        bs = settings.BATCH_SIZE
        max_iterations = settings.MAX_ITERATIONS
        display_step = settings.LOG_INTERVAL

        coord = tf.train.Coordinator()
        queue = FlowQueue(fts, coord)

        X = queue.dequeue_many(bs)

        net = load_net()
        net.init(**dict(zip(fts.fields, X)))

        with tf.Session() as sess:
            queue.start(sess)

            net.begin(sess)

            try:
                step = 0
                while not coord.should_stop() and not net.should_stop():

                    net.train(sess)

                    if step % display_step == 0:
                        net.display(sess, step)

                    step += 1
            except KeyboardInterrupt:
                pass

            coord.request_stop()
            queue.stop()
            coord.join(stop_grace_period_secs=5)


