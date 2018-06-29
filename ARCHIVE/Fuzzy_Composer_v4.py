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

# GAUSSIAN
from ASAMs.GAUSSIAN.GaussianSAM_ps import GaussianSAM as gsam_ps
from ASAMs.GAUSSIAN.GaussianSAM_pnr import GaussianSAM as gsam_pnr
from ASAMs.GAUSSIAN.GaussianSAM_snn import GaussianSAM as gsam_snn

# LAPLACE
from ASAMs.LAPLACE.LaplaceSAM_ps import LaplaceSAM as lsam_ps
from ASAMs.LAPLACE.LaplaceSAM_pnr import LaplaceSAM as lsam_pnr
from ASAMs.LAPLACE.LaplaceSAM_snn import LaplaceSAM as lsam_snn

# TRIANGLE
from ASAMs.TRIANGLE.TriangleSAM_ps import TriangleSAM as trisam_ps
from ASAMs.TRIANGLE.TriangleSAM_pnr import TriangleSAM as trisam_pnr
from ASAMs.TRIANGLE.TriangleSAM_snn import TriangleSAM as trisam_snn

import os
import pickle
from six.moves.urllib.request import urlopen

import distutils.dir_util as util
import numpy as np
import tensorflow as tf
import scipy.io as spio
import UTIL.ChordSymbolsLib as chords_lib

################################################# Global Variables #####################################################

# MODEL SHAPES AND SAVE NAMES
gauss = 0
laplace = 1
tri = 2
names = ['gaussian', 'laplace', 'triangle']

# MODEL SAVE INFO
trial_tag = ''
type_tag = ''
bases = []
save_model = []
models = []


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
    bases = []

    # SETUP DIRECTORY
    for i in range(len(names)):
        # print(i)
        path = output_folder + '/' + names[i]
        util.mkpath(path)
        bases.append(path + '/' + type_str)
        util.mkpath(bases[i])
        bases[i] = bases[i] + '/TRIAL' + trial_num
        util.mkpath(bases[i])
        util.mkpath(bases[i] + '/Saved_Models')

######################################### Data Collection and Organization #############################################


# Retrieves the data set.
def get_data_set():
    filename = cwd + '/DATA_SETS/cross_ent_data_set.pkl'
    with open(filename,'rb') as handle: #Get data set
        data = pickle.load(handle)
    if data is not None:
        print('DATA SET: RETRIEVED\n')
    else:
        print('DATA SET: MALFORMED, NO DATA RETRIEVED\n')
    return data


# Splits training and evaluation data
def split_data():  
    chords_train = []
    melodies_train = []
    chords_eval = []
    melodies_eval = []
    train_indices = []
    eval_indices = []
    m = 'melody' # Melody column in data_set value
    c = 'chords' # Chord progression column in data_set value
    files = list(data_set.keys())
        
    num_files = len(files)
    
    # EVAL DATA
    if len(files) > 1:
        for i in range(num_eval):
            print('ORGANIZING EVAL DATA . . .')
        
            # Determine index to pick
            eval_index = int(num_files * np.random.random()) #Random index selection
            while eval_index in eval_indices: #No duplicates
                eval_index = int(num_files * np.random.random()) 
        
            eval_indices.append(eval_index)
            eval_song = files[eval_index] #Grab the corresponding song
        
            # Add to melody and chords
            melodies_eval.append(data_set[eval_song][m]) 
            chords_eval.append(data_set[eval_song][c])
        
    # TRAINING DATA
    for i in range(num_train):
        print('ORGANIZING TRAINING DATA . . .')
        
        #Determine index to pick
        train_index = int(num_files*np.random.random())
        while train_index in train_indices or train_index in eval_indices: #No duplicates between training and eval indices AND within training indices
            train_index = int(num_files*np.random.random())
            
        train_indices.append(train_index)
        train_song = files[train_index] #Grab the corresponding song
        
        #Add to melody and chords
        melodies_train.append(data_set[train_song][m])
        chords_train.append(data_set[train_song][c])
        
    #print('Eval Indices: ', eval_indices)
    #print('Training Indices: ', train_indices)
    
    return melodies_train, chords_train, melodies_eval, chords_eval


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


# Trains the fuzzy systems.
def train():
    # TRAIN MODEL
    pnr = 0
    snn = 1
    ps = 2
    model = 0
    model_file = 1
    asam_types = ['Pitched Note or Rest', 'Sustain or New Note', 'Pitch Selector']
    gauss_models = [[gs_pnr, 'gsam_pnr.ckpt'],
                    [gs_snn, 'gsam_snn.ckpt'],
                    [gs_ps, 'gsam_ps.ckpt']]

    # laplace_models = [[ls_pnr, 'lsam_pnr.ckpt'],
    #                 [ls_snn, 'lsam_snn.ckpt'],
    #                 [ls_ps, 'lsam_ps.ckpt']]

    # tri_models = [[ts_pnr, 'trisam_pnr.ckpt'],
    #                 [ts_snn, 'trisam_snn.ckpt'],
    #                 [ts_ps, 'trisam_ps.ckpt']]

    asams = [gauss_models]
    for run in range(1):
        errors = np.zeros([len(asams), len(asam_types)])
        melodies_train, chords_train, melodies_eval, chords_eval = split_data()
        melodies_all, all_chords = format_melodies(melodies_train, chords_train)
        print(melodies_train[0])
        print(melodies_all[ps][0])
        print(len(melodies_all[ps][0]))
        print(all_chords[0])
        print(len(all_chords[0]))
        print('MELODY TRAINING, RUN ', run + 1, ' . . .')
        for type in range(1):  # (len(asams)):
            models = asams[type]
            filename = bases[type] + '/' + names[type] + type_tag + trial_tag
            print('TRAINING WITH: ', names[type], ' . . .')
            for i in range(len(models)):
                print(asam_types[i])
                save_model[i] = bases[type] + '/Saved_Models/' + asams[type][i][model_file]
                errors[type][i] = models[i][model].train(melodies_all[i], adapt_iters, lr, epoch_size, save_model[i],
                                                         filename)
            # models[i].train(melodies_train, adapt_iters, lr, epoch_size, save_model[i], filename, run)

        for type in range(len(asam_types)):
            min_error = min(errors[type])
            val = None
            i = -1
            while not val == min_error:
                i += 1
                val = errors[type][i]
            print(asam_types[type], ' Minimum error: ', val, ' w/ ', names[i])


# Generates melodies from the trained fuzzy systems given primer notes and a test chord progression
def gen_melody():
    test_chords = ['Cmaj7', 'Am7', 'Dm7', 'G7', 'Cmaj7', 'Am7', 'Dm7', 'G7']
    num_notes = 16
    the_chords = []
    for c, chord in enumerate(test_chords):
        for n in range(num_notes):
            the_chords.append(chords_lib.chord_symbol_pitches(chord))
    num_measures = len(test_chords)
    num_repeats = 4
    npm = 16
    octave = 4
    oct_range = 4
    primer_notes = [60, 66, 67]

    for k in range(1):  # (len(models)):
        test_pitches = models[k].pitches_given_chords(the_chords, primer_notes, save_model[k])

        # OUTPUT NOTE MATRIX
        nm_filename = bases[k] + '/' + names[k] + type_tag + trial_tag + '_melody_gen.mat'
        create_note_matrix(test_pitches, nm_filename)


def main():
    # DEFINE PARAMS
    epoch_size = 10  # REPORT RESULTS AFTER epoch_size ITERATIONS
    num_rules = 15  # NUM RULES
    adapt_iters = epoch_size * 10  # NUMBER OF TRAINING ITERATIONS
    lr = 0.0001  # LEARNING RATE

    # GRAB TRAINING DATA
    num_train = 50
    num_eval = 10
    cwd = ''
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
    features = 5

    # GAUSSIAN
    gs_pnr = gsam_pnr(num_rules, features, memory)
    gs_snn = gsam_snn(num_rules, features, memory)
    gs_ps = gsam_ps(num_rules, features, memory)

    # LAPLACE
    # ls_pnr = lsam_pnr(num_rules, features, memory)
    # ls_snn = lsam_snn(num_rules, features, memory)
    # ls_ps = lsam_ps(num_rules, features, memory)

    # TRIANGLE
    # ts_pnr = trisam_pnr(num_rules, features, memory)
    # ts_snn = trisam_snn(num_rules, features, memory)
    # ts_ps = trisam_ps(num_rules, features, memory)


    # for i in range(len(shapes)):
    #    util.mkpath(output_folder + '/' + shapes[i][model_name])

    ans = str(raw_input('Train (y/n)? '))
    if ans == 'y':
        train()

    ans = str(raw_input('Create Music (y/n)? '))
    if ans == 'y':
        gen_melody()


if __name__ == '__main__': main()

