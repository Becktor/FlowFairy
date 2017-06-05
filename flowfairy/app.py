import tensorflow as tf
import numpy as np
import itertools as it
import importlib

from flowfairy.conf import settings
from flowfairy.utils import take
from flowfairy import data
from flowfairy.feature import FeatureManager
from flowfairy.core.queue import FlowQueue
from flowfairy.core.stage import stage


def load_net():
    net = importlib.import_module(settings.NET).Net()
    return net

global_step = 0

def set_global_step(s):
    global global_step
    global_step = s

def run(*args, **options):

    coord = tf.train.Coordinator()
    net = load_net()

    queues = []
    with tf.variable_scope('network') as scope:
        for data_loader in data.provider:
            fts = FeatureManager(data_loader)
            queue = FlowQueue(fts, coord)
            queues.append(queue)


            with tf.name_scope(data_loader.name):
                X = queue.dequeue()
                func = getattr(net, data_loader.name)
                func(**dict(zip(fts.fields, X)))
            scope.reuse_variables()

    with tf.Session() as sess:

        stage.before(sess, net)
        for queue in queues: queue.start(sess)

        sess.run(tf.global_variables_initializer())

        try:
            step = global_step + 1
            while not coord.should_stop() and not net.should_stop():

                stage.run(sess, step)

                step += 1
        except KeyboardInterrupt:
            pass

        coord.request_stop()
        queue.stop()
        coord.join(stop_grace_period_secs=5)
