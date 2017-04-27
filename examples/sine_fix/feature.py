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

class FrequencyGen(Feature):

    def feature(self):
        return {'frequency': frqs[frq], 'frequency_id': np.array(frq, dtype=np.int32)}

    class Meta:
        ignored_fields = ('frequency',)

class SineGen(Feature):
    arr = np.arange(samplerate, dtype=np.float32) * 2 * np.pi
    frqs = np.arange(frq_min, frq_max, step) / samplerate
    choices = np.arange(frequency_count)

    def feature(self, **kwargs):
        choice = np.random.choice(self.choices, size=(2,1), replace=False)

        sines = np.tile(self.arr, (2,1))
        chosen_frqs = self.frqs[choice]

        x = np.sin(sines * chosen_frqs).astype('float32')
        y = x[0]
        x = x.sum(axis=0)

        return {'y': y, 'x': x, 'frqid': np.array(chosen_frqs[0], dtype=np.int32)}


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
        return {'x': classify(x), 'y': classify(y)}
