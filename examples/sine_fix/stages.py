import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from flowfairy.core.stage import register, Stage
from flowfairy.conf import settings


@register(10000)
class SummaryStage(Stage):
    def fig2rgb_array(self, expand=True):
        self.figure.canvas.draw()
        buf = self.figure.canvas.tostring_rgb()
        ncols, nrows = self.figure.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
        return np.fromstring(buf, dtype=np.uint8).reshape(shape)

    def reset_fig(self):
        self.figure = plt.figure(num=0, figsize=(6,4), dpi=300)
        self.figure.clf()

    def before(self, sess, net):
        tf.summary.scalar('loss', net.loss)

        self.pred = net.pred

        self.reset_fig()
        img = self.fig2rgb_array()

        with tf.variable_scope("graph"):
            image = tf.get_variable("graph_image", shape=img.shape, dtype=tf.uint8)
            sess.run(image.initializer)

        tf.summary.image('graph', image)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(settings.LOG_DIR, str(datetime.now())), sess.graph)

    def plot(self, sess):
        self.reset_fig()

        res = sess.run(self.pred)

        start = 0
        end = 200

        plt.subplot('111').plot(res[0,start:end],'r')
        #plt.subplot('111').plot(batch_y[-1,start:end],'b', alpha=0.5)
        #plt.subplot('111').plot(batch_x[-1,start:end],'g', alpha=0.5)


    def draw_img(self, sess):
        self.plot(sess)
        with tf.variable_scope("graph", reuse=True):
            image = tf.get_variable("graph_image", dtype=tf.uint8)
            sess.run(image.assign(self.fig2rgb_array()))

    def run(self, sess, i):
        self.draw_img(sess)

        summary = sess.run(self.merged)

        self.writer.add_summary(summary, i)



@register
class TrainingStage(Stage):

    def before(self, sess, net):
        self.optimizer = net.optimizer

    def run(self, sess, i):
        sess.run(self.optimizer)
