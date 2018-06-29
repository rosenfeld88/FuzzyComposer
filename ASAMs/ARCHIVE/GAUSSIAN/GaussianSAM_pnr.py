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

    # GAUSSIAN STANDARD ADDITIVE MODEL (SAM)
    # AUTHOR: TAYLOR ROSENFELD
    # PARTNER: YAN ZHU
    # START DATE: 11/12/17

class GaussianSAM:
    
    #num_rules: number of rules for SAM (user-defined)
    def __init__(self, num_rules, num_feats, memory):
        self.num_rules = num_rules
        num_labels = 128
        self.num_features = num_feats
        self.memory_size = memory
        #SET UP MODEL PARAMETERS
        self.mem_wgts = tf.Variable(tf.truncated_normal([memory, num_rules], mean = 1, stddev = 2))
        self.m = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.d = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.w = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.v = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.c = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.cw = tf.Variable(tf.truncated_normal([num_labels], mean = 1, stddev = 2))
           
    #WRITES DATA TO OUTPUT FILE    
    #def __write_data(self, error, filename):
    #    f = open(filename + '_error.mat', 'w')
    #    f.write('Epoch,Error\n')
    #    for i in range(len(error)):
    #        f.write(str(x_data[i]) + ',' + str(y_data[i]) + ',' + str(error[i]) + '\n')
            
    def __get_octave(self, pitch):
        return int(pitch/12)
             
    def train(self, melodies, adapt_iters, lr, epoch_size, model_save, filename):
        # PREPARE TRAINING DATA
        conditioner = tf.placeholder(tf.float32, shape = None)
        actual_note = tf.placeholder(tf.int32, shape = None)
        
        # FUZZY APPROX
        x = tf.tensordot(conditioner, self.mem_wgts, 1)
        ax = tf.exp(tf.multiply(-0.5, tf.square(tf.divide(tf.subtract(x, self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.c), tf.multiply(self.v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.v), ax))
        fuzzy_approx = tf.divide(num, den)
        learn_note = tf.nn.softmax(tf.multiply(self.cw, tf.clip_by_value(fuzzy_approx, 0, 1)))
        #tf.cast(learn_note, tf.int32)
        
        # DEFINE LOSS, TRAIN, AND SAVE OPS
        #loss = learn_interval - feature
        #loss = tf.reduce_mean(tf.square(learn_interval - feature))
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = actual_note, logits = learn_note))
        #train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(lr, 0.9, 0.999).minimize(cross_entropy)
        #train_op = tf.train.AdagradOptimizer(lr).minimize(loss)
        saver = tf.train.Saver({'m': self.m, 'd': self.d, 'w': self.w, 'v': self.v, 'c': self.c, 'cw': self.cw})
        
        #CREATE SESSION AND INITIALIZE
        sess = tf.Session()        
        sess.run(tf.global_variables_initializer())
            
        #TRAINING LOOP
        #error = 0
        epoch_ent = np.zeros(adapt_iters)
        for epoch in range(adapt_iters):
            cross_ent = 0
            for m, melody in enumerate(melodies): #For each song
                song_ent = 0
                start = self.memory_size
                for n in range(start, len(melody)): #For each note in the song
                    note = melody[n]
                    features = []
                    #previous = melody[n-1]
                    for i in range(self.memory_size):
                        prev = melody[n-i-1]
                        #print(prev)
                        prev_oct = self.__get_octave(prev)
                        features.append(prev)
                        #for j in range(self.num_features - 1):
                        #    features.append(np.array(chord[j]) + (prev_oct * 12))
                    #assert features not np.any(np.isnan(features))
                    if note + 2 > 129 or note + 2 < 0:
                        print('OH SHIT')
                    _, run_ent = sess.run([train_op, cross_entropy], feed_dict = {conditioner: features, actual_note: note})
                    #print(sess.run([learn_note], feed_dict = {conditioner: features, actual_note: int(melody[n] + 2)}))
                    #print(run_ent)
                    song_ent += run_ent
                    
                cross_ent += song_ent/(len(melody) - 3)
                
            epoch_ent[epoch] = cross_ent/len(melodies)
            print(epoch_ent[epoch])
                
            if epoch % epoch_size == 0.0:
                print("Training Step: " + str(epoch))
        
        spio.savemat(filename + '_pnr_error.mat', mdict = {'error': epoch_ent})
        saver.save(sess, model_save)
            
        sess.close()
        
        return epoch_ent[adapt_iters - 1]
        
    def generate(self, chord_prog, primer_notes, model_save):
        num_notes = len(chord_prog)
        #PREPARE INPUT AND OUTPUT
        conditioner = tf.placeholder(tf.float32, shape = None)
        x = tf.tensordot(conditioner, self.mem_wgts, 1)
        
        #SYSTEM
        ax = tf.exp(tf.multiply(-0.5, tf.square(tf.divide((x - self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.c), tf.multiply(self.v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.v), ax))
        fuzzy_approx = tf.divide(num, den)
        fuzzy_approx = tf.clip_by_value(fuzzy_approx, 0, 1)
        learn_note = tf.nn.softmax(tf.multiply(self.cw, fuzzy_approx))
        learn_note = tf.argmax(tf.cast(learn_note, tf.int32))
        
        #SAVER
        saver = tf.train.Saver({'m': self.m,
                                'd': self.d,
                                'w': self.w,
                                'v': self.v,
                                'c': self.c,
                                'cw': self.cw,
                                'mem_wgts': self.mem_wgts})
        
        #CREATE AND RUN SESSION
        melody = [] 
        with tf.Session() as sess:
            saver.restore(sess, model_save)
            for n, note in enumerate(primer_notes):
                melody.append(note)
            start = self.memory_size
            #print(melody)
            for m in range(start, num_notes):
                previous = []
                for i in range(self.memory_size):
                    prev = melody[m-i-1]
                    previous.append(prev)
        
                melody.append(sess.run(learn_note, feed_dict = {conditioner: previous}))
                    
        return melody
                
    def rhythm_given_pitch(self, pitches, npm, num_measures, num_repeats, model_save):
        rhythm = {}
    
        #PREPARE INPUT AND OUTPUT
        pitch = tf.placeholder(tf.float32, shape = None)
    
        #SYSTEM
        ax = tf.exp(tf.multiply(-0.5,tf.square(tf.divide((pitch-self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.c),tf.multiply(self.v,ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.v),ax))
        sample = tf.divide(num,den)
    
        #SAVER
        saver = tf.train.Saver()
        
        #CREATE AND RUN SESSION 
        with tf.Session() as sess:
            saver.restore(sess, model_save)
            meas = 1
            for r in range(num_repeats):
                for m in range(num_measures):
                    measure = np.zeros(npm)
                    for n in range(npm):
                        measure[n] = sess.run(sample, feed_dict = {pitch: pitches[meas][n]})
                    rhythm[meas] = measure
                    meas += 1        
        return rhythm
        
    
                
        

        
        

        
             
        