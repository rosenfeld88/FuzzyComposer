from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf
import UTIL.ChordSymbolsLib as chords_lib
import scipy.io as spio


## FUZZY MUSIC COMPOSITION ##

# Pitched Note or Rest Model Class
# Supports Laplace, Gaussian, or Triangular rules
# AUTHOR: TAYLOR ROSENFELD
# START DATE: 6/26/18

class Model:

    # num_rules: number of rules for ASAM
    # num_feats: number of features to learn
    # memory: size of memory
    # asam_type: ASAM shape (Gaussian, Laplace, Triangle)
    # labels: number of labels for classification
    # model_filename: Filename for model save
    # error_filename: Filename for saving error (cross entropy) data
    def __init__(self, num_rules, num_feats, memory, asam_type, labels, model_filename, error_filename):
        self.__num_rules = num_rules # Number of rules for ASAM
        self.__shape = asam_type # Rule shape (Gaussian, Laplace, etc.)
        self.__num_labels = labels # Number of labels for classification
        self.__num_features = num_feats
        self.__memory_size = memory
        self.__model_fname = model_filename
        self.__error_fname = error_filename
        self.__error = 0

        # SET UP MODEL PARAMETERS
        self.__mem_wgts = tf.Variable(tf.truncated_normal([memory, num_rules], mean = 0, stddev = 1))
        self.__m = tf.Variable(tf.truncated_normal([self.__num_rules]))
        self.__d = tf.Variable(tf.truncated_normal([self.__num_rules]))
        self.__w = tf.Variable(tf.truncated_normal([self.__num_rules]))
        self.__v = tf.Variable(tf.truncated_normal([self.__num_rules]))
        self.__c = tf.Variable(tf.truncated_normal([self.__num_rules]))
        self.__cw = tf.Variable(tf.truncated_normal([self.__num_labels], mean = 0, stddev = 1))

    def __get_octave(self, pitch):
        return int(pitch / 12)

    def get_shape(self):
        return self.__shape

    def get_error_file(self):
        return self.__error_fname

    def get_model_file(self):
        return self.__model_fname

    def get_error(self):
        return self.__error

    def train(self, melodies, adapt_iters, lr, epoch_size):
        # PREPARE TRAINING DATA
        conditioner = tf.placeholder(tf.float32, shape = None)
        actual_note = tf.placeholder(tf.int32, shape = None)

        # FUZZY APPROX
        x = tf.tensordot(conditioner, self.__mem_wgts, 1)
        ax = []
        if self.__shape == 'gaussian':
            ax = tf.exp(tf.multiply(-0.5, tf.square(tf.divide(tf.subtract(x, self.__m), self.__d))))
        elif self.__shape == 'laplace':
            ax = tf.exp(tf.multiply(-1.0, tf.abs(tf.divide((x - self.__m), self.__d))))
        elif self.__shape == 'triangle':
            ax = tf.subtract(1.0, tf.divide(tf.abs(x - self.__m), self.__d))
            ax = tf.clip_by_value(ax, 0.0, 99999)

        num = tf.reduce_sum(tf.multiply(tf.multiply(self.__w, self.__c), tf.multiply(self.__v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.__w, self.__v), ax))
        fuzzy_approx = tf.divide(num, den)
        learn_note = tf.nn.softmax(tf.multiply(self.__cw, tf.clip_by_value(fuzzy_approx, 0, 1)))

        # DEFINE LOSS, TRAIN, AND SAVE OPS
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actual_note, logits = learn_note))
        train_op = tf.train.AdamOptimizer(lr, 0.9, 0.999).minimize(cross_entropy)
        saver = tf.train.Saver({'m': self.__m, 'd': self.__d, 'w': self.__w, 'v': self.__v, 'c': self.__c, 'cw': self.__cw})

        # CREATE SESSION AND INITIALIZE
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # TRAINING LOOP
        epoch_ent = np.zeros(adapt_iters)
        for epoch in range(adapt_iters):
            cross_ent = 0
            for m, melody in enumerate(melodies):  # For each song
                song_ent = 0
                start = self.__memory_size
                for n in range(start, len(melody)):  # For each note in the song
                    note = melody[n]
                    features = []
                    for i in range(self.__memory_size):
                        prev = melody[n - i - 1]
                        prev_oct = self.__get_octave(prev)
                        features.append(prev)
                    if note + 2 > 129 or note + 2 < 0:
                        print('OH SHIT')
                    _, run_ent = sess.run([train_op, cross_entropy],
                                          feed_dict = {conditioner: features, actual_note: note})
                    song_ent += run_ent

                cross_ent += song_ent / (len(melody) - 3)

            epoch_ent[epoch] = cross_ent / len(melodies)

            if epoch % epoch_size == 0.0:
                print("Training Step: " + str(epoch))
                print("Cross Entropy: {}".format(epoch_ent[epoch]))

        spio.savemat(self.__error_fname, mdict = {'error': epoch_ent})
        #spio.savemat(filename + '_pnr_error.mat', mdict = {'error': epoch_ent})
        saver.save(sess, self.__model_fname)

        sess.close()

        self.error = epoch_ent[adapt_iters - 1]

    def generate(self, chord_prog, primer_notes):
        num_notes = len(chord_prog)
        # PREPARE INPUT AND OUTPUT
        conditioner = tf.placeholder(tf.float32, shape = None)
        x = tf.tensordot(conditioner, self.mem_wgts, 1)

        # SYSTEM
        ax = tf.exp(tf.multiply(-0.5, tf.square(tf.divide((x - self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.c), tf.multiply(self.v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.v), ax))
        fuzzy_approx = tf.divide(num, den)
        fuzzy_approx = tf.clip_by_value(fuzzy_approx, 0, 1)
        learn_note = tf.nn.softmax(tf.multiply(self.cw, fuzzy_approx))
        learn_note = tf.argmax(tf.cast(learn_note, tf.int32))

        # SAVER
        saver = tf.train.Saver({'m': self.m,
                                'd': self.d,
                                'w': self.w,
                                'v': self.v,
                                'c': self.c,
                                'cw': self.cw,
                                'mem_wgts': self.mem_wgts})

        # CREATE AND RUN SESSION
        melody = []
        with tf.Session() as sess:
            saver.restore(sess, self.model_fname)
            for n, note in enumerate(primer_notes):
                melody.append(note)
            start = self.memory_size
            # print(melody)
            for m in range(start, num_notes):
                previous = []
                for i in range(self.memory_size):
                    prev = melody[m - i - 1]
                    previous.append(prev)

                melody.append(sess.run(learn_note, feed_dict = {conditioner: previous}))

        return melody









