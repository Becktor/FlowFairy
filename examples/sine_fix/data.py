import numpy as np
import random
import itertools as it
import os
import soundfile as sf
import glob

from flowfairy import data
from flowfairy.conf import settings

"""
#samplerate = settings.SAMPLERATE
#frequency_count = settings.CLASS_COUNT
#frq_min, frq_max = settings.FREQUENCY_LIMIT
#step = (frq_max - frq_min) / frequency_count
#
#frqs = list(enumerate(np.arange(frq_min, frq_max, step)))
#blnds = list(frqs)
##random.shuffle(blnds)
#
##val_cut = int(frequency_count * 0.75)
#
#def frequencies():
    #while True:
        #random.shuffle(frqs)
        #yield from frqs
#
#def blends_train():
    #blends = blnds[0::4] + blnds[1::4] + blnds[2::4]
    #while True:
        #random.shuffle(blends)
        #yield from blends
#
#def blends_val():
    #blends = blnds[3::4]
    #while True:
        #random.shuffle(blends)
        #yield from blends
#
#data.register(frequencies, 'train')
#data.register(frequencies, 'validation')
#
#data.register(blends_train, 'train', name='blends')
#data.register(blends_val, 'validation', name='blends')
"""

basedir = '/home/sorson/audio_dataset/preprocessed_22050/'

def speakers(subdir):
    filenames = list(glob.glob(os.path.join(basedir, subdir, '*.npz')))
    def shuffler():
        while True:
            random.shuffle(filenames)
            yield from filenames
    return shuffler


data.register(speakers('train'), 'train', name='speaker')
data.register(speakers('dev'), 'validation', name='speaker')

data.register(speakers('train'), 'train', name='blend')
data.register(speakers('dev'), 'validation', name='blend')
