import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from flowfairy.core.stage import register, Stage
from flowfairy.conf import settings

def norm(val):
    val = (val-tf.minimum(val))/(tf.maximum(val)-tf.minimum(val))
    return val

@register(500)
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
        tf.summary.scalar('acc', net.accuracy)
        tf.summary.scalar('cost', net.cost)
        self.pred = net.pred
        self.x = net.x
        self.y = net.y
        tf.summary.audio('x', self.x, settings.SAMPLERATE)
        tf.summary.audio('pred', self.pred, settings.SAMPLERATE)
        self.chunk=net.chunk
        self.reset_fig()
        img = self.fig2rgb_array()
        self.image = tf.Variable(np.zeros(img.shape, dtype=np.uint8))

        tf.summary.image('graph', self.image)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(settings.LOG_DIR, str(datetime.now())), sess.graph)

    def plot(self, sess):
        self.reset_fig()

        res, x, y, c = sess.run([self.pred, self.x, self.y ,self.chunk])
        res = np.argmax(res, 2)
        start = c[0]-50
        end = (start+settings.CHUNK+100)
        plt.subplot('111').plot(res[0,start:end],'r')
        plt.subplot('111').plot(y[0,start:end],'b', alpha=0.5)
        plt.subplot('111').plot(x[0,start:end],'g', alpha=0.5)


    def draw_img(self, sess):
        self.plot(sess)
        sess.run(self.image.assign(self.fig2rgb_array()))

    def run(self, sess, i):
        self.draw_img(sess)

        summary = sess.run(self.merged)

        self.writer.add_summary(summary, i)



@register()
class TrainingStage(Stage):

    def before(self, sess, net):
        self.optimizer = net.optimizer

    def run(self, sess, i):
        sess.run(self.optimizer)
