import tensorflow as tf
import numpy as np
import os
import io
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from feature import classify

import matplotlib.pyplot as plt

from flowfairy.core.stage import register, Stage, stage
from flowfairy.conf import settings


def get_log_dir():
    return os.path.join(settings.LOG_DIR, settings.LOGNAME)

@register(250)
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
        tf.summary.scalar('train_acc', net.train_acc)
        tf.summary.scalar('train_cost', net.train_cost)
        tf.summary.scalar('val_acc', net.val_acc)
        tf.summary.scalar('val_cost', net.val_cost)
        tf.contrib.layers.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.net = net

        self.reset_fig()
        img = self.fig2rgb_array()

        self.train_image_in = tf.placeholder(np.uint8, shape=img.shape)
        self.train_image = tf.Variable(np.zeros(img.shape, dtype=np.uint8), trainable=False, name='train_graph_image')
        self.train_image_assign = self.train_image.assign(self.train_image_in)
        tf.summary.image('train_graph', self.train_image)

        self.val_image_in = tf.placeholder(np.uint8, shape=img.shape)
        self.val_image = tf.Variable(np.zeros(img.shape, dtype=np.uint8), trainable=False, name='val_graph_image')
        self.val_image_assign = self.val_image.assign(self.val_image_in)
        tf.summary.image('val_graph', self.val_image)

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(get_log_dir(), sess.graph)

    def plot(self, sess, pred, x, y):
        self.reset_fig()

        res, x, y = sess.run([ pred, x, y ])
        x = classify(x)
        res = np.argmax(res, 2)

        start = np.random.randint(500)
        end = start + 128

        plt.subplot('111').plot(res[0,start:end],'r')
        plt.subplot('111').plot(y[0,start:end],'b', alpha=0.5)
        plt.subplot('111').plot(x[0,start:end],'g', alpha=0.5)


    def draw_img(self, sess):
        self.plot(sess, self.net.train_pred, self.net.train_x, self.net.train_y)
        sess.run(self.train_image_assign, feed_dict={self.train_image_in: self.fig2rgb_array()})

        self.plot(sess, self.net.val_pred, self.net.val_x, self.net.val_y)
        sess.run(self.val_image_assign, feed_dict={self.val_image_in: self.fig2rgb_array()})

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


@register(1000)
class SavingStage(Stage):
    def before(self, sess, net):
        self.saver = tf.train.Saver()

    def run(self, sess, i):
        self.saver.save(sess, get_log_dir()+'.ckpt', global_step=i, latest_filename=settings.LOGNAME+'.checkpoint')
