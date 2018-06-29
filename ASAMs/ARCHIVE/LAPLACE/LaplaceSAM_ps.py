from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf
import UTIL.ChordSymbolsLib as chords_lib
import scipy.io as spio

##FUZZY MUSIC COMPOSITION##

    #GAUSSIAN STANDARD ADDITIVE MODEL (SAM)
    #AUTHOR: TAYLOR ROSENFELD
    #PARTNER: YAN ZHU
    #START DATE: 11/12/17

class LaplaceSAM:
    
    #num_rules: number of rules for SAM (user-defined)
    def __init__(self, num_rules, memory):
        self.num_rules = num_rules
        num_labels = 130
        #SET UP MODEL PARAMETERS
        self.mem_wgts = tf.Variable(tf.truncated_normal([memory]))
        self.m = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.d = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.w = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.v = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.c = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.cw = tf.Variable(tf.truncated_normal([num_labels]))
           
    #WRITES DATA TO OUTPUT FILE    
    def __write_data(self, filename, note, cond, F_meas ,error):
        f = open(filename, 'w') 
        f.write('Conditioner,Output,loss\n') 
        for i in range(len(note[cond])):
            f.write(str(x_data[i]) + ',' + str(y_data[i]) + ',' + str(error[i]) + '\n')
            
    def __get_octave(self, pitch):
        return int(pitch/12)
             
    def train(self, melodies, chords, adapt_iters, lr, epoch_size, model_save, filename, run):
        #PREPARE TRAINING DATA
        conditioner = tf.placeholder(tf.float32, shape = None)
        feature = tf.placeholder(tf.int32, shape = None)
        
        #FUZZY APPROX
        x = tf.tensordot(conditioner, self.mem_wgts, 1)
        ax = tf.exp(tf.multiply(-1.0, tf.abs(tf.divide((x - self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.c), tf.multiply(self.v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.v), ax))
        fuzzy_approx = tf.divide(num, den)
        learn_note = tf.nn.softmax(tf.multiply(self.cw, fuzzy_approx))
        tf.cast(learn_note, tf.int32)
        
        #DEFINE LOSS, TRAIN, AND SAVE OPS
        #loss = learn_interval - feature
        #loss = tf.reduce_mean(tf.square(learn_interval - feature))
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = feature, logits = learn_note)
        #train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(lr, 0.9, 0.999).minimize(cross_entropy)
        #train_op = tf.train.AdagradOptimizer(lr).minimize(loss)
        saver = tf.train.Saver({'m': self.m, 'd': self.d, 'w': self.w, 'v': self.v, 'c': self.c, 'cw': self.cw, 'mem_wgts': self.mem_wgts})
        
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
                chord_prog = chords[m]
                start = 3
                for n in range(start, len(melody)): #For each note in the song
                    note = melody[n]
                   
                    prev1 = melody[n-1]
                    prev2 = melody[n-2]
                    prev3 = melody[n-3]
                    
                    octave1 = self.__get_octave(prev1) #Get octave of previous note
                    octave2 = self.__get_octave(prev2)
                    octave3 = self.__get_octave(prev3)
                    
                    r1 = chord_prog[n - 1] + (octave1 * 12)
                    if r1 > prev1:
                        r1 -= 12
                    r2 = chord_prog[n - 2] + (octave2 * 12)
                    if r2 > prev2:
                        r2 -= 12
                    r3 = chord_prog[n - 3] + (octave3 * 12)
                    if r3 > prev3:
                        r3 -= 12
                        
                    memory = [prev1 - r1, prev2 - r2, prev3 - r3]
            
                    _, run_ent = sess.run([train_op, cross_entropy], feed_dict = {conditioner: memory, feature: int(melody[n] + 2)})
                    song_ent += run_ent
                    
                cross_ent += song_ent/(len(melody) - 3)
                
            epoch_ent[epoch] = cross_ent/len(melodies)
                
            if epoch % epoch_size == 0.0:
                print("Training Step: " + str(epoch))
        
        spio.savemat(filename + '_run_' + str(run) + '_error.mat', mdict = {'error': epoch_ent})
        saver.save(sess, model_save)
            
        sess.close()
        
        return epoch_ent[adapt_iters - 1]
        
    def pitches_given_chords(self, chords, primer_notes, model_save):
        pitches = {}
        num_notes = len(chords)
        #PREPARE INPUT AND OUTPUT
        conditioner = tf.placeholder(tf.float32, shape = None)
        x = tf.tensordot(conditioner, self.mem_wgts, 1)
        
        #SYSTEM
        ax = tf.exp(tf.multiply(-1.0, tf.abs(tf.divide((x - self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.c),tf.multiply(self.v,ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.v),ax))
        fuzzy_approx = tf.divide(num, den)
        learn_note = tf.nn.softmax(tf.multiply(self.cw, fuzzy_approx))
        learn_note = tf.argmax(tf.cast(learn_note, tf.int32))
        
        #SAVER
        saver = tf.train.Saver({'m': self.m, 'd': self.d, 'w': self.w, 'v': self.v, 'c': self.c, 'cw': self.cw, 'mem_wgts': self.mem_wgts})
        
        #CREATE AND RUN SESSION
        melody = [] 
        with tf.Session() as sess:
            saver.restore(sess, model_save)
            for n, note in enumerate(primer_notes):
                melody.append(note)
            start = 3
            #print(melody)
            for i in range(start, num_notes):
                #print(melody)
                #print(i)
                chord = chords[i]
                #print(chord)
                prev1 = melody[i-1]
                #print(prev1)
                prev2 = melody[i-2]
                prev3 = melody[i-3]
                
                octave1 = self.__get_octave(prev1) #Get octave of previous note
                octave2 = self.__get_octave(prev2)
                octave3 = self.__get_octave(prev3)
                
                r1 = chords[i - 1] + (octave1 * 12)
                if r1 > prev1:
                    r1 -= 12
                r2 = chords[i - 2] + (octave2 * 12)
                if r2 > prev2:
                    r2 -= 12
                r3 = chords[i - 3] + (octave3 * 12)
                if r3 > prev3:
                    r3 -= 12
                    
                memory = [prev1 - r1, prev2 - r2, prev3 - r3]
        
                melody.append(sess.run(learn_note, feed_dict = {conditioner: memory}) - 2)
                    
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
        
    
                
        

        
        

        
             
        