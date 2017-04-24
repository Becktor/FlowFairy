import tensorflow as tf
import threading

from flowfairy.conf import settings

class FlowQueue:

    def __init__(self, features, coordinator):
        self.coord = coordinator
        self.features = features

        capacity = settings.QUEUE_CAPACITY

        with tf.name_scope("queue"):
            self.queue = tf.PaddingFIFOQueue(capacity, features.dtypes, shapes=features.shapes)
            self.enqueue_op = self.queue.enqueue(features.placeholders)

    def _enqueue(self):
        while not self.coord.should_stop():
            batch = next(self.features)
            self.sess.run(self.enqueue_op, feed_dict=dict(zip(self.features.placeholders, batch)))

    def start(self, sess):
        num_threads = settings.QUEUE_NUM_THREADS

        self.sess = sess

        for _ in range(num_threads):
            t = threading.Thread(target=self._enqueue)
            self.coord.register_thread(t)
            t.daemon = True
            t.start()

    def dequeue(self):
        return self.queue.dequeue()

    def dequeue_many(self, batch_size):
        return self.queue.dequeue_many(batch_size)
