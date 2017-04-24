import tensorflow as tf
import numpy as np
import itertools as it
import threading
import importlib

from flowfairy.conf import settings
from flowfairy.utils import take
from flowfairy.data.loader import DataLoader
from flowfairy.feature import FeatureManager

def setup_queue(features, capacity=128):

    with tf.name_scope("queue"):
        print("shapes", features.shapes)
        queue = tf.PaddingFIFOQueue(capacity, features.dtypes, shapes=features.shapes)
        enqueue_op = queue.enqueue(features.placeholders)

        return queue, enqueue_op

def enqueue(coord, sess, enqueue_op, features):
    while not coord.should_stop():
        batch = next(features)
        sess.run(enqueue_op, feed_dict=dict(zip(features.placeholders, batch)))

def start_queue(coord, sess, enqueue_op, features, num_threads):
    for _ in range(num_threads):
        t = threading.Thread(target=enqueue, args=(coord, sess, enqueue_op, features))
        coord.register_thread(t)
        t.daemon = True
        t.start()

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
        queue, enqueue_op = setup_queue(fts)

        X = queue.dequeue_many(bs)

        net = load_net()
        net.init(**dict(zip(fts.fields, X)))

        with tf.Session() as sess:
            start_queue(coord, sess, enqueue_op, fts, 1)

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
            queue.close(cancel_pending_enqueues=True)
            coord.join(stop_grace_period_secs=5)


