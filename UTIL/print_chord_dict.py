from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from GaussianSAM_v7 import GaussianSAM
from TanhSAM import TanhSAM
from CauchySAM import CauchySAM
from TriangleSAM_v2 import TriangleSAM
from SincSAM import SincSAM
from LaplaceSAM_v2 import LaplaceSAM

import os
import pickle
from six.moves.urllib.request import urlopen

import distutils.dir_util as util
import numpy as np
import tensorflow as tf
import scipy.io as spio
import ChordSymbolsLib as chords_lib

cwd = os.getcwd()
filename = cwd + '/DATA_SETS/chord_dict.pkl'
with open(filename, 'rb') as handle:
    chord_dict = pickle.load(handle)
    
#print(chord_dict)

print(np.array(chords_lib.chord_symbol_pitches('Am7(b9)')) + 12*5)