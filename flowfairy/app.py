import tensorflow as tf
import numpy as np
import itertools as it
import importlib

from flowfairy.conf import settings
from flowfairy.utils import take
from flowfairy.data.loader import DataLoader
from flowfairy.feature import FeatureManager
from flowfairy.core.queue import FlowQueue
from flowfairy.core.stage import stage


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

            stage.before(sess, net)
            queue.start(sess)

            #net.begin(sess)

            sess.run(tf.global_variables_initializer())

            try:
                step = 1
                while not coord.should_stop() and not net.should_stop():

                    stage.run(sess, step)

                    step += 1
            except KeyboardInterrupt:
                pass

            coord.request_stop()
            queue.stop()
            coord.join(stop_grace_period_secs=5)
