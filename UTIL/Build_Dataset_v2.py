from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from MusicXMLParser import MusicXMLDocument
from MusicXMLParser import ChordSymbol
import ChordSymbolsLib as chord_lib
from MusicXMLParser import MusicXMLParseException
from fractions import Fraction
import xml.etree.ElementTree as ET
import zipfile
import pickle

# internal imports

import six
from magenta.music import constants

import os
from six.moves.urllib.request import urlopen

import numpy as np
import scipy.io as spio
import tensorflow as tf

#EXTRACT CHORDS, CHORD MEASURES, AND MEASURES
def get_chord_prog(chords):
    chord_prog = []
    num_notes = 16
    for c, chord in enumerate(chords):
        for n in range(num_notes):
            #chord_prog.append(chord_lib.chord_symbol_root(chord))
            chord_prog.append(chord_lib.chord_symbol_pitches(chord))
    return chord_prog

#EXTRACT MELODIES FROM MEASURES
def get_melody(note_matrix):
    melody = []
    start = 0
    dur = 1
    channel = 2
    pitch = 3
    velocity = 4
    rest = -2
    sustain = -1
    note_res = 0.25
    
    for n, note in enumerate(note_matrix):
        note_dur = note[dur]
        note_pitch = note[pitch]
        note_start = note[start]
        melody.append(note_pitch)
        if note_dur > note_res:
            num_hold = int(note_dur/note_res)
            for i in range(num_hold - 1):
                melody.append(note_pitch)
    
        if n < len(note_matrix) - 1:
            next_note = note_matrix[n + 1]
            rest_dur = next_note[start] - (note[start] + note_dur)
            if not rest_dur == 0.0:
                rest_bins = int(rest_dur/note_res)
                for i in range(rest_bins):
                    melody.append(note_pitch)
                 
    return melody
    

#CREATE THE DATA SET AND CHORD DICTIONARIES  
def gen_data(folders, directory, chord_progs):
    all_data = {}
    num_fucked = 0
    for fold, folder in enumerate(folders):
        data = {}
        path = directory + folder
        files = os.listdir(path)
        for f, the_file in enumerate(files):
            print(the_file)
            if the_file.endswith('.mat'):
                filename = path + '/' + the_file
                print(filename)
                matfile = {}
                spio.loadmat(filename, mdict = matfile)
                nm = matfile['nm']
                melody = get_melody(nm)
                chords = chord_progs[fold]
                chord_prog = get_chord_prog(chords)
                data['melody'] = melody
                data['chords'] = chord_prog
                if not len(melody) == len(chord_prog):
                    num_fucked += 1
                    print('SKIPPED: ', the_file, ', MALFORMED MELODY')
                    #print('FUCK ME')
                else:
                    all_data[the_file] = data
        
    print('NUMBER FUCKED UP: ', num_fucked)
    
    return all_data    

#SAVE FILE VIA PICKLE
def save_data_set(data, data_name):
    with open(data_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
#SETTINGS/SETUP
FOUR_FOUR = '4/4'
cwd = os.getcwd() #Working directory 
data_dir = cwd + '/DATA/generate_11chords/' #Directory holding MusicXML files
all_folders = os.listdir(data_dir) #All filenames
progs = []
progs.append(['Am7(b5)', 'D7', 'Gm7', 'C7', 'Am7(b5)', 'D7', 'Gm7', 'C7'])
progs.append(['Bm7(b5)', 'E7(#9)', 'Am7', 'Am7', 'Bm7(b5)', 'E7(#9)', 'Am7', 'Am7'])
progs.append(['Cm7', 'Ebm6', 'Bbmaj7', 'Bbmaj7', 'Cm7', 'Ebm6', 'Bbmaj7', 'Bbmaj7'])
progs.append(['Cmaj7', 'Dm7', 'Em7', 'Dm7', 'Cmaj7', 'C#dim7', 'Dm7', 'G7'])
progs.append(['Ebmaj7', 'Gbdim7', 'Fm7', 'Bb7', 'Ebmaj7', 'Gbdim7', 'Fm7', 'Bb7'])
progs.append(['Em7', 'Am7', 'Dm7', 'G7', 'Em7', 'Am7', 'Dm7', 'G7'])
progs.append(['F7', 'Eb7', 'Bb7', 'Bb7', 'F7', 'Eb7', 'Bb7', 'Bb7'])
progs.append(['Fm7', 'Am7', 'Fm7', 'Dm7', 'Fm7', 'Am7', 'Fm7', 'Dm7'])
progs.append(['Fmaj7', 'F#dim7', 'Gm7', 'C7', 'Am7', 'Dm7', 'Gm7', 'C7'])
progs.append(['Gm7', 'C7', 'Cm7', 'F7', 'Gm7', 'C7', 'Cm7', 'F7'])
progs.append(['Gmaj7', 'Bb7', 'Ebmaj7', 'Abmaj7', 'Am7', 'D7', 'Gmaj7', 'Dm7'])
data_set = gen_data(all_folders, data_dir, progs)
#print(data_set['chords'])
#print(data_set['melody'])
#print(len(data_set['melody']) == len(data_set['chords']))
print('TOTAL NUMBER OF SONGS: ', len(data_set),'\n')
data_set_dir = cwd + '/DATA_SETS'
#os.mkdir(data_set_dir)
data_set_name = data_set_dir + '/cross_ent_data_set.pkl'
#chord_dict_name = data_set_dir + '/attya_chords.pkl'
save_data_set(data_set, data_set_name)
#save_data_set(chord_dict, chord_dict_name)


