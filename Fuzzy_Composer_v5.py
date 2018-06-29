# Fuzzy Composer
#
# Description: This is a program for training a fuzzy system to create melodies given chord progressions.
# Training is based on MusicXML files from the Wikifonia database. Melodies are represented as arrays with 1/16 note
# resolution. Each element of the array is an integer from -2 to 127. Values from 0 to 127 represent standard MIDI
# pitches, while -1 indicates a note sustain event and -2 indicates a rest.
#
# The learning task is divided into three decision types: pitched note or rest (pnr), sustain or new note (snn), and
# pitch selection (ps). Melody arrays described above are transformed into three intermediate arrays for training. The
# pnr and snn arrays are binary while the ps array elements can vary from 0 to 127 (MIDI pitches). New and sustained
# notes correspond, respectively, to 1s and 0s in the snn array, while pitched notes and rests correspond, respectively,
# to 1s and 0s in the pnr array. Binary classification is used to train the snn and pnr ASAMs while k-in-1
# classification is used for the ps ASAMs.
#
# AUTHOR: TAYLOR ROSENFELD
# PARTNER: YAN ZHU
# START DATE: 11/12/17

##################################################### Imports ##########################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ASAM IMPORTS
from ASAMs.Model import Model


import os
import pickle
from six.moves.urllib.request import urlopen

import distutils.dir_util as util
import numpy as np
import tensorflow as tf
import scipy.io as spio
import UTIL.ChordSymbolsLib as chords_lib

################################################# Global Variables #####################################################

# SYSTEM PARAMETERS
epoch_size = 10  # REPORT RESULTS AFTER epoch_size ITERATIONS
num_rules = 15  # NUM RULES
adapt_iters = epoch_size * 10  # NUMBER OF TRAINING ITERATIONS
lr = 0.0001  # LEARNING RATE
num_train = 50
#num_eval = 10

# MODEL SHAPES AND SAVE NAMES
gauss = 0
laplace = 1
tri = 2
shapes = ['gaussian', 'laplace', 'triangle']
pnr_models = []
snn_models = []
ps_models = []

# MODEL SAVE INFO AND FILING
trial_tag = ''
type_tag = ''
bases = {}
save_model = []
models = []
cwd = ''

# DATA
data_set = {}


################################################# Utility Methods ######################################################

# Method for writing data to file.
# @param filename Name of file to which data will be written.
# @param in_data input data
# @param out_data output data
def write_data(filename, in_data, out_data):
    with open(filename, 'w') as handle:
        handle.write('input,output\n')
        for i in range(len(in_data)):
            handle.write(str(in_data[i]) + ',' + str(out_data[i]) + '\n')


# Method for creating a MIDI note matrix from a melody.
# @param melody Array containing melody notes to be written to note matrix
# @param midi_filename Filename to write.
def create_note_matrix(melody, midi_filename):
    # GENERATE NOTE MATRIX
    rest = -2 # rest label
    sustain = -1 # sustain label
    note_res = 0.25 # 1/16 note (1/4 note = 1)
    nm_re = []
    note_dur = note_res
    ch = 1
    vel = 64
    last_added = -1
    last_pitch = -1
    beat = 0
    prev_start = 0
    for n, note in enumerate(melody):
        if note == sustain:
            note_dur += note_res
            nm_re[last_added] = [prev_start, note_dur, ch, last_pitch, vel]
            beat += note_res
        elif note == rest:
            beat += note_res
        else:
            last_added += 1
            last_pitch = note
            if not beat == 0:
                beat += note_res
            nm_re.append([beat, note_dur, ch, note, vel])
            prev_start = beat
            note_dur = note_res

    # SAVE TO MAT FILE
    spio.savemat(midi_filename, mdict = {'nm': nm_re})


# Sets up working directory
def filing_setup():
    global type_tag, trial_tag, bases, cwd

    # SETUP OUTPUT FOLDER
    cwd = os.getcwd()  # STAY GLOBAL
    output_folder = cwd + '/OUTPUT'
    util.mkpath(output_folder)

    type_str = str(raw_input('Enter experiment type: ')).lower()
    type_tag = '_' + type_str
    trial_num = str(raw_input('Enter trial number: '))
    trial_tag = '_T' + trial_num
    base = output_folder + '/' + type_str
    util.mkpath(base)
    # SETUP DIRECTORY
    for shape in shapes:
        path = base + '/' + shape
        util.mkpath(path)
        path = path + '/TRIAL' + trial_num
        util.mkpath(path)
        bases[shape] = path
        path = path + '/Saved_Models'
        util.mkpath(path)

######################################### Data Collection and Organization #############################################


# Retrieves the data set.
def get_data_set():
    filename = cwd + '/DATA_SETS/cross_ent_data_set.pkl'
    with open(filename,'rb') as handle:  # Get data set
        data = pickle.load(handle)
    if data is not None:
        print('DATA SET: RETRIEVED\n')
    else:
        print('DATA SET: MALFORMED, NO DATA RETRIEVED\n')
    return data


# Splits training and evaluation data
def split_data():
    global num_train
    chords_train = []
    melodies_train = []
    # chords_eval = []
    # melodies_eval = []
    train_indices = []
    # eval_indices = []
    m = 'melody' # Melody column in data_set value
    c = 'chords' # Chord progression column in data_set value
    files = list(data_set.keys())
        
    num_files = len(files)
    
    # EVAL DATA
    # if len(files) > 1:
    #     for i in range(num_eval):
    #         print('ORGANIZING EVAL DATA . . .')
    #
    #         # Determine index to pick
    #         eval_index = int(num_files * np.random.random()) #Random index selection
    #         while eval_index in eval_indices: #No duplicates
    #             eval_index = int(num_files * np.random.random())
    #
    #         eval_indices.append(eval_index)
    #         eval_song = files[eval_index] #Grab the corresponding song
    #
    #         # Add to melody and chords
    #         melodies_eval.append(data_set[eval_song][m])
    #         chords_eval.append(data_set[eval_song][c])
        
    # TRAINING DATA
    for i in range(num_train):
        print('ORGANIZING TRAINING DATA . . .')

        # Determine index to pick
        train_index = int(num_files * np.random.random())
        while train_index in train_indices: # No duplicates
            train_index = int(num_files * np.random.random())

        train_indices.append(train_index)
        train_song = files[train_index] # Grab the corresponding song

        # Add to melody and chords
        melodies_train.append(data_set[train_song][m])
        chords_train.append(data_set[train_song][c])
        
    #print('Eval Indices: ', eval_indices)
    #print('Training Indices: ', train_indices)
    
    #return melodies_train, chords_train, melodies_eval, chords_eval
    return melodies_train, chords_train


# Formats melodies for use with the various ASAMs.
def format_melodies(melodies, chord_progs):
    pnr_melodies = []
    snn_melodies = []
    ps_melodies = []
    all_chord_progs = []
    pitched_note = 1
    rest = 0
    new_note = 1
    sus_note = 0
    prog = 0
    for melody in melodies:
        chord_prog = chord_progs[prog]
        pnr_mel = []
        snn_mel = []
        ps_mel = []
        chords = []
        for n in range(len(melody)):
            note = melody[n]
            if note >= -1:
                pnr_mel.append(pitched_note)
                if note == -1:
                    snn_mel.append(sus_note)
                else:
                    snn_mel.append(new_note)
                    ps_mel.append(note)
                    chords.append(chord_prog[n])
            else:
                pnr_mel.append(rest)

        pnr_melodies.append(pnr_mel)
        snn_melodies.append(snn_mel)
        ps_melodies.append(ps_mel)
        all_chord_progs.append(chords)
        prog += 1
    return [pnr_melodies, snn_melodies, [ps_melodies, all_chord_progs]]

############################################## Model Training and Evaluation ###########################################


# # Trains the fuzzy systems.
# def train():
#     melodies_train, chords_train = split_data()
#     melodies_all, all_chords = format_melodies(melodies_train, chords_train)
#
#     print('PITCHED NOTE/REST TRAINING . . .')
#     for pnr_model in pnr_models:  # (len(asams)):
#         print('TRAINING WITH: ', pnr_model.get_shape(), ' . . .')
#         pnr_model.train(melodies_all[0], adapt_iters, lr, epoch_size)
#
#     print('SUSTAIN/NEW NOTE TRAINING . . .')
#     for snn_model in snn_models:  # (len(asams)):
#         print('TRAINING WITH: ', snn_model.get_shape(), ' . . .')
#         snn_model.train(melodies_all[1], adapt_iters, lr, epoch_size)
#
#     print('PITCH SELECTOR TRAINING . . .')
#     for ps_model in ps_models:  # (len(asams)):
#         print('TRAINING WITH: ', ps_model.get_shape(), ' . . .')
#         ps_model.train(melodies_all[2], adapt_iters, lr, epoch_size)
#
#     for type in range(len(asam_types)):
#         min_error = min(errors[type])
#         val = None
#         i = -1
#         while not val == min_error:
#             i += 1
#             val = errors[type][i]
#         print(asam_types[type], ' Minimum error: ', val, ' w/ ', names[i])


# Trains PNR models
def train_pnr(melodies):
    print('PITCHED NOTE/REST TRAINING . . .')
    for pnr_model in pnr_models:  # (len(asams)):
        print('TRAINING WITH: ', pnr_model.get_shape(), ' . . .')
        pnr_model.train(melodies, adapt_iters, lr, epoch_size)

    min_error_val = pnr_models[0].get_error()
    min_error_model = pnr_models[0].get_shape()
    for i in range(1, len(pnr_models) - 1):
        model_error = pnr_models[i].get_error()
        if model_error < min_error_val:
            min_error_val = model_error
            min_error_model = pnr_models[i].get_shape()

    print('Pitched Note/Rest Min Error: ', min_error_val, ' w/ ', min_error_model)


# Trains SNN models
def train_snn(melodies):
    print('SUSTAIN/NEW NOTE TRAINING . . .')
    for snn_model in snn_models:  # (len(asams)):
        print('TRAINING WITH: ', snn_model.get_shape(), ' . . .')
        snn_model.train(melodies, adapt_iters, lr, epoch_size)

    min_error_val = snn_models[0].get_error()
    min_error_model = snn_models[0].get_shape()
    for i in range(1, len(snn_models) - 1):
        model_error = snn_models[i].get_error()
        if model_error < min_error_val:
            min_error_val = model_error
            min_error_model = snn_models[i].get_shape()

    print('Sustain/New Note Min Error: ', min_error_val, ' w/ ', min_error_model)


# Trains PS models
def train_ps(data):
    print('SUSTAIN/NEW NOTE TRAINING . . .')
    for ps_model in ps_models:  # (len(asams)):
        print('TRAINING WITH: ', ps_model.get_shape(), ' . . .')
        ps_model.train(data, adapt_iters, lr, epoch_size)

    min_error_val = ps_models[0].get_error()
    min_error_model = ps_models[0].get_shape()
    for i in range(1, len(ps_models) - 1):
        model_error = ps_models[i].get_error()
        if model_error < min_error_val:
            min_error_val = model_error
            min_error_model = ps_models[i].get_shape()

    print('Pitch Selector Min Error: ', min_error_val, ' w/ ', min_error_model)


# Generates melodies from the trained fuzzy systems given primer notes and a test chord progression
# def gen_melody():
#     test_chords = ['Cmaj7', 'Am7', 'Dm7', 'G7', 'Cmaj7', 'Am7', 'Dm7', 'G7']
#     num_notes = 16
#     the_chords = []
#     for c, chord in enumerate(test_chords):
#         for n in range(num_notes):
#             the_chords.append(chords_lib.chord_symbol_pitches(chord))
#     num_measures = len(test_chords)
#     num_repeats = 4
#     npm = 16
#     octave = 4
#     oct_range = 4
#     primer_notes = [60, 66, 67]
#
#     for k in range(1):  # (len(models)):
#         test_pitches = models[k].pitches_given_chords(the_chords, primer_notes, save_model[k])
#
#         # OUTPUT NOTE MATRIX
#         nm_filename = bases[k] + '/' + names[k] + type_tag + trial_tag + '_melody_gen.mat'
#         create_note_matrix(test_pitches, nm_filename)


def main():
    global pnr_models, snn_models, ps_models, data_set

    # Setup Filing
    filing_setup()

    # GRAB TRAINING DATA
    data_set = get_data_set()

    # FEATURES
    # note_start = 0 #Note start (in beats)
    # note_dur = 1 #Note duration (in beats)
    # channel = 2 #MIDI Channel (most likely 1, by default)
    # pitch = 3 #Note pitch (0-127)
    # velocity = 4 #Note velocity (most likely 64, by default)
    # active_chord = 5 #"Active" chord over note

    # CREATE AND INITIALIZE SAMs
    memory = 3
    num_feats = 5
    err_ext = '_error.mat'
    # GAUSSIAN
    gauss_file = 'gaussian' + type_tag + trial_tag
    gs_pnr = Model(num_rules, num_feats, memory, 'gaussian', 2, gauss_file + '_pnr.ckpt', gauss_file + '_pnr' + err_ext)
    gs_snn = Model(num_rules, num_feats, memory, 'gaussian', 2, gauss_file + '_snn.ckpt', gauss_file + '_snn' + err_ext)
    gs_ps = Model(num_rules, num_feats, memory, 'gaussian', 128, gauss_file + '_ps.ckpt', gauss_file + '_ps' + err_ext)

    # LAPLACE
    # laplace_file = 'laplace' + type_tag + trial_tag
    # lp_pnr = Model(num_rules, num_feats, memory, 'laplace', 2, laplace_file + '_pnr.ckpt', laplace_file + '_pnr' + err_ext)
    # lp_snn = Model(num_rules, num_feats, memory, 'laplace', 2, laplace_file + '_snn.ckpt', laplace_file + '_snn' + err_ext)
    # lp_ps = Model(num_rules, num_feats, memory, 'laplace', 128, laplace_file + '_ps.ckpt', laplace_file + '_ps' + err_ext)

    # TRIANGLE
    # tri_file = 'triangle' + type_tag + trial_tag
    # tri_pnr = Model(num_rules, num_feats, memory, 'triangle', 2, tri_file + '_pnr.ckpt', tri_file + '_pnr' + err_ext)
    # tri_snn = Model(num_rules, num_feats, memory, 'triangle', 2, tri_file + '_snn.ckpt', tri_file + '_snn' + err_ext)
    # tri_ps = Model(num_rules, num_feats, memory, 'triangle', 128, tri_file + '_ps.ckpt', tri_file + 'ps' + err_ext)

    melodies_train, chords_train = split_data()
    training_music = format_melodies(melodies_train, chords_train)

    pnr_models = [gs_pnr]
    snn_models = [gs_snn]
    ps_models = [gs_ps]

    ans = str(raw_input('Train (y/n)? '))
    if ans == 'y':
        train_pnr(training_music[0])
        train_snn(training_music[1])
        train_ps(training_music[2])

    # ans = str(raw_input('Create Music (y/n)? '))
    # if ans == 'y':
    #     gen_melody()


if __name__ == '__main__': main()

