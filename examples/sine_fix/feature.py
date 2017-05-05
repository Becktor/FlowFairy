from flowfairy.feature import Feature
from flowfairy.conf import settings

import numpy as np


samplerate = settings.SAMPLERATE
duration = settings.DURATION
frequency_count = settings.CLASS_COUNT
frq_min, frq_max = settings.FREQUENCY_LIMIT
step = (frq_max - frq_min) / frequency_count
sine_count = settings.SINE_COUNT

def classify(val):
    val = (val-np.min(val))/(np.max(val)-np.min(val))
    return np.floor(val*255)

class SineGen(Feature):
    arr = np.arange(samplerate, dtype=np.float32) * 2 * np.pi / samplerate

    def feature(self, initial, **kwargs):
        frq1, frq2 = initial

        sines = np.tile(self.arr, (2,1))

        x = np.sin(sines * np.array([[ frq1[1] ], [frq2[1]]])).astype('float32')
        y = x[0]
        x = x.sum(axis=0)

        return {'cls': y, 'y': y, 'x': x, 'frqid': np.array(frq1[0], dtype=np.int32)}

    class Meta:
        ignored_fields = ('initial',)


class NoisySineGen(Feature):

    def feature(self, x, **kwargs):
        noise = np.random.uniform(-0.5, 0.5, samplerate).astype('float32')
        return {'x': noise+x}


class Mask(Feature):

    def feature(self, **kwargs):
        return {'m': np.ones(samplerate * duration, dtype=np.float32)}


class Dropout(Feature):

    def feature(self, **kwargs):
        return {'keep_prob': np.array(0.50, dtype=np.float32)}


class ConvertToClasses(Feature):

    def feature(self, x, y, **kwargs):
        return {'x': classify(x), 'y': classify(y).astype('int64')}
