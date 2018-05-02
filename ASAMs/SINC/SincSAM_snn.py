from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf
import ChordSymbolsLib as chords_lib

##FUZZY MUSIC COMPOSITION##

    #GAUSSIAN STANDARD ADDITIVE MODEL (SAM)
    #AUTHOR: TAYLOR ROSENFELD
    #PARTNER: YAN ZHU
    #START DATE: 11/12/17

class SincSAM:
    
    #num_rules: number of rules for SAM (user-defined)
    def __init__(self, num_rules):
        self.num_rules = num_rules
        
        #SET UP MODEL PARAMETERS
        self.m = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.d = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.w = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.v = tf.Variable(tf.truncated_normal([self.num_rules]))
        self.c = tf.Variable(tf.truncated_normal([self.num_rules]))
           
    #WRITES DATA TO OUTPUT FILE    
    def __write_data(self, filename, note, cond, F_meas ,error):
        f = open(filename, 'w') 
        f.write('Conditioner,Output,loss\n') 
        for i in range(len(note[cond])):
            f.write(str(x_data[i]) + ',' + str(y_data[i]) + ',' + str(error[i]) + '\n')
            
    def __get_octave(self, pitch):
        return int(pitch/12)
             
    def train(self, melodies, adapt_iters, lr, epoch_size, filename, feat, cond, savename, do_print):
        #PREPARE TRAINING DATA
        conditioner = tf.placeholder(tf.float32, shape = None)
        feature = tf.placeholder(tf.float32, shape = None)
        
        #FUZZY APPROX
        xmd = tf.divide((conditioner - self.m), self.d)
        ax = tf.divide(tf.sin(xmd), xmd)
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.c), tf.multiply(self.v, ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w, self.v), ax))
        learn_interval = tf.divide(num, den)
        
        #DEFINE LOSS, TRAIN, AND SAVE OPS
        loss = tf.reduce_mean(tf.square(learn_interval - feature))
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        #train_op = tf.train.AdamOptimizer().minimize(loss)
        saver = tf.train.Saver()
        
        #CREATE SESSION AND INITIALIZE
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        #TRAINING LOOP
        error = 0
        for epoch in range(adapt_iters):
            for s in range(len(melodies)): #For each song
                song_error = 0
                mel = melodies[s]
                prev_note = None
                num_notes = 0
                for m, measure in mel.iteritems(): #For each measure in the song
                    i = 0
                    start = 0
                    if m == 1:
                        start = 1
                        prev_note = measure[0]
                        i += 1
                    for n in range(start, len(measure)): #For each note in the song
                        note = measure[n]
                        octave = self.__get_octave(prev_note[feat]) #Get octave of previous note
                        chord = prev_note[cond] #Get active chord on previous note
                        root = chords_lib.chord_symbol_root(chord) + (octave * 12) #Transpose root of previous note chord to correct octave
                        interval = prev_note[feat] - root #Determine function of previous note (relative to chord root)
                        sess.run(train_op, feed_dict = {conditioner: interval, feature: note[feat] - prev_note[feat]})
                        song_error += sess.run(loss, feed_dict = {conditioner: interval, feature: note[feat] - prev_note[feat]})
                        i += 1
                        prev_note = note
                        if do_print:
                            print('Chord: ', note[cond], ' Expected Pitch: ', note[feat])
                            print('Denominator: ', deno)
                        num_notes += 1
                    
                error += song_error/num_notes
                
            if epoch % epoch_size == 0.0:
                print("Training Step: " + str(epoch))
        
        saver.save(sess, savename)
            
        sess.close()
        
        return error/len(melodies)
        
    def pitches_given_chords(self, chords, npm, num_measures, num_repeats, primer_note, oct_range, savename):
        pitches = {}
        
        #PREPARE INPUT AND OUTPUT
        prev = tf.placeholder(tf.float32, shape = None)
        
        #SYSTEM
        ax = tf.exp(tf.multiply(-0.5,tf.square(tf.divide((prev-self.m), self.d))))
        num = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.c),tf.multiply(self.v,ax)))
        den = tf.reduce_sum(tf.multiply(tf.multiply(self.w,self.v),ax))
        learn_interval = tf.divide(num,den)
        
        #SAVER
        saver = tf.train.Saver()
        
        #CREATE AND RUN SESSION 
        with tf.Session() as sess:
            num_notes = npm * num_measures * num_repeats
            saver.restore(sess, savename)
            meas = 1
            prev_note = primer_note
            cen_oct = self.__get_octave(primer_note)
            for r in range(num_repeats): 
                for m in range(num_measures):
                    measure = np.zeros(npm)
                    start = 0
                    if m == 0 and r == 0:
                        measure[0] = prev_note
                        start = 1
                    for n in range(start, npm):
                        chord = chords[m]
                        print(chord)
                        octave = self.__get_octave(prev_note)
                        root = chords_lib.chord_symbol_root(chord) + (12 * octave)
                        interval = prev_note - root
                        chosen_interval = sess.run(learn_interval, feed_dict = {prev: interval})
                        measure[n] = prev_note + int(chosen_interval)
                        if self.__get_octave(measure[n]) > cen_oct + (oct_range/2):
                            measure[n] -= 12 * (self.__get_octave(measure[n]) - cen_oct)
                        elif self.__get_octave(measure[n]) < cen_oct - (oct_range/2):
                            measure[n] += 12 * (cen_oct - self.__get_octave(measure[n]))
                        print(measure[n])
                        prev_note = measure[n]
                    pitches[meas] = measure 
                    meas += 1   
                    
        return pitches
                
    def rhythm_given_pitch(self, pitches, npm, num_measures, num_repeats, savename):
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
            saver.restore(sess, savename)
            meas = 1
            for r in range(num_repeats):
                for m in range(num_measures):
                    measure = np.zeros(npm)
                    for n in range(npm):
                        measure[n] = sess.run(sample, feed_dict = {pitch: pitches[meas][n]})
                    rhythm[meas] = measure
                    meas += 1        
        return rhythm
        
    
                
        

        
        

        
             
        