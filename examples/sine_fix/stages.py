import tensorflow as tf
import os
from datetime import datetime

from flowfairy.core.stage import register, Stage
from flowfairy.conf import settings


@register(10)
class SummaryStage(Stage):

    def before(self, sess, net):
        tf.summary.scalar('loss', net.loss)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(settings.LOG_DIR, str(datetime.now())), sess.graph)

    def run(self, sess, i):
        summary = sess.run([self.merged])

        print("adding summary")



        self.writer.add_summary(summary, i)

        print("done summary")
