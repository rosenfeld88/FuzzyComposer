from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf
import UTIL.ChordSymbolsLib as chords_lib
import scipy.io as spio
from ASAMs.Model import Model


## FUZZY MUSIC COMPOSITION ##

# Pitched Note or Rest Model Class
# Supports Laplace, Gaussian, or Triangular rules
# AUTHOR: TAYLOR ROSENFELD
# START DATE: 6/26/18

class PitchSelectorModel(Model):
    def train(self, data, adapt_iters, lr, epoch_size):
        mel = 0
        ch = 1
        melodies = data[mel]
        chords = data[ch]
        # PREPARE TRAINING DATA
        conditioner = tf.placeholder(tf.float32, shape = None)
        actual_note = tf.placeholder(tf.int32, shape = None)

        # FUZZY APPROX
        x = tf.tensordot(conditioner, self.__mem_wgts, 1)
        ax = tf.exp(tf.multiply(-0.5, tf.square(tf.divide(tf.subtract(x, self.__m), self.__d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.__w, self.__c), tf.multiply(self.__v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.__w, self.__v), ax))
        fuzzy_approx = tf.divide(num, den)
        learn_note = tf.nn.softmax(tf.multiply(self.__cw, tf.clip_by_value(fuzzy_approx, 0, 127)))
        tf.cast(learn_note, tf.int32)

        # DEFINE LOSS, TRAIN, AND SAVE OPS
        # loss = learn_interval - feature
        # loss = tf.reduce_mean(tf.square(learn_interval - feature))
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actual_note, logits = learn_note))
        # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(lr, 0.9, 0.999).minimize(cross_entropy)
        # train_op = tf.train.AdagradOptimizer(lr).minimize(loss)
        saver = tf.train.Saver({'m': self.__m,
                                'd': self.__d,
                                'w': self.__w,
                                'v': self.__v,
                                'c': self.__c,
                                'cw': self.__cw,
                                'mem_wgts': self.__mem_wgts})

        # CREATE SESSION AND INITIALIZE
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # TRAINING LOOP
        # error = 0
        epoch_ent = np.zeros(adapt_iters)
        for epoch in range(adapt_iters):
            cross_ent = 0
            for m, melody in enumerate(melodies):  # For each song
                song_ent = 0
                chord_prog = chords[m]
                start = self.__memory_size
                for n in range(start, len(melody)):  # For each note in the song
                    note = melody[n]
                    features = []
                    previous = melody[n - 1]
                    for i in range(self.__memory_size):
                        chord = chord_prog[i]
                        prev = melody[n - i - 1]
                        # print(prev)
                        prev_oct = self.__get_octave(prev)
                        features.append(prev)
                        # for j in range(self.num_features - 1):
                        #    features.append(np.array(chord[j]) + (prev_oct * 12))
                    # assert features not np.any(np.isnan(features))
                    if note + 2 > 129 or note + 2 < 0:
                        print('OH SHIT')
                    _, run_ent = sess.run([train_op, cross_entropy],
                                          feed_dict = {conditioner: previous + 2, actual_note: note + 2})
                    # print(sess.run([learn_note], feed_dict = {conditioner: features, actual_note: int(melody[n] + 2)}))
                    # print(run_ent)
                    song_ent += run_ent

                cross_ent += song_ent / (len(melody) - 3)

            epoch_ent[epoch] = cross_ent / len(melodies)
            print(epoch_ent[epoch])

            if epoch % epoch_size == 0.0:
                print("Training Step: " + str(epoch))

        spio.savemat(self.__error_fname, mdict = {'error': epoch_ent})
        saver.save(sess, self.__model_fname)

        sess.close()

        self.__error = epoch_ent[adapt_iters - 1]