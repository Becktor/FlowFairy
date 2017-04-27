from flowfairy.feature import Feature
from flowfairy.conf import settings

import numpy as np


samplerate = settings.SAMPLERATE
duration = settings.DURATION


class FrequencyGen(Feature):

    def feature(self):
        frqs = np.random.randint(340, 720)
        return {'frequency': frqs}

def classify(val):
    val = (val-np.min(val))/(np.max(val)-np.min(val))
    return np.floor(val*255)

class SineGen(Feature):
    arr = np.arange(samplerate, dtype=np.float32) * 2 * np.pi

    def feature(self, frequency, **kwargs):
        return {'y':np.sin(self.arr * (frequency)/samplerate)}

    def fields(self):
        return ('y',)

class NoisySineGen(Feature):

    def feature(self, y, **kwargs):
        noise = np.random.uniform(-0.5, 0.5, samplerate).astype('float32')
        return {'x': noise+y}

    def fields(self):
        return ('x',)

class LabelGen(Feature):

    def feature(self, x, y, **kwargs):
        return {'x':classify(x), 'y':classify(y)}

class Mask(Feature):

    def feature(self, **kwargs):
        return {'m': np.ones(samplerate * duration, dtype=np.float32)}

    def fields(self):
        return ('m',)



class Dropout(Feature):

    def feature(self, **kwargs):
        return {'keep_prob': np.array(0.50, dtype=np.float32)}

    def fields(self):
        return ('keep_prob',)