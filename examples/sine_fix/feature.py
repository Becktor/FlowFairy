from flowfairy.feature import Feature
from flowfairy.conf import settings

import numpy as np


samplerate = settings.OUTPUTLEN + 2000
duration = settings.DURATION
frequency_count = settings.CLASS_COUNT
frq_min, frq_max = settings.FREQUENCY_LIMIT
step = (frq_max - frq_min) / frequency_count
max_amp = settings.MAX_AMP

def classify(val):
    val = (val-np.min(val))/(np.max(val)-np.min(val))
    return np.floor(val*255)

class SineGen(Feature):
    arr = np.arange(samplerate, dtype=np.float32) * 2 * np.pi / samplerate

    def feature(self, frequencies, blends, **kwargs):
        frq1, frq2 = frequencies, blends

        amp = np.random.rand(2,1) * max_amp + 1 # [1;max_amp)
        phase = np.random.rand(2,1) * np.pi * 2
        sines = np.tile(self.arr, (2,1)) * amp

        x = (np.sin(sines * np.array([[ frq1[1] ], [frq2[1]]]) + phase) * amp).astype('float32')
        x = x[:,:,None] # add channel
        y = x[0]

        return {'y': y, 'x': x[0], 'blend': x[1], 'frqid': np.array(frq1[0], dtype=np.int32), 'frqid2': np.array(frq2[0])}

    class Meta:
        ignored_fields = ('frequencies', 'blends')

class NoisySineGen(Feature):

    def feature(self, x, blend, **kwargs):
        noise = np.random.uniform(-0.5, 0.5, (2, samplerate, 1)).astype('float32')
        return {'x': noise[0]+x, 'blend': blend+noise[1]}


class Mask(Feature):

    def feature(self, **kwargs):
        return {'m': np.ones(samplerate * duration, dtype=np.float32)}


class Dropout(Feature):

    def feature(self, **kwargs):
        return {'keep_prob': np.array(0.50, dtype=np.float32)}


class ConvertToClasses(Feature):

    def feature(self, x, y, **kwargs):
        return {'y': classify(y).astype('int64')}
