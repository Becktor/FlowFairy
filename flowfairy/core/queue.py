import tensorflow as tf
import itertools as it
import threading

from flowfairy.conf import settings

def _batch_shapes(shapes, bs):
    return [[bs] + list( shape ) for shape in shapes]

class FlowQueue:

    def __init__(self, features, coordinator):
        self.coord = coordinator
        self.features = features
        self.bs = settings.BATCH_SIZE

        self._create_placeholders()

        capacity = settings.QUEUE_CAPACITY
        with tf.name_scope("queue"):
            self.queue = tf.FIFOQueue(capacity, features.dtypes, shapes=_batch_shapes(features.shapes, self.bs))
            self.enqueue_op = self.queue.enqueue(self._placeholders)


    def _create_placeholders(self):
        self._placeholders = [
            tf.placeholder(dtype, shape, name=name) for dtype, shape, name in zip(
                self.features.dtypes,
                _batch_shapes(self.features.shapes, self.bs),
                self.features.fields
            )
        ]

    def _enqueue(self):
        while not self.coord.should_stop():
            batch = self._batch()
            self.sess.run(self.enqueue_op, feed_dict=dict(zip(self._placeholders, batch)))

    def _batch(self):
        return list(zip(*list(it.islice(self.features, self.bs)))) # Take bs from generator

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
        return self.queue.dequeue()

    def stop(self):
        self.queue.close(cancel_pending_enqueues=True)
