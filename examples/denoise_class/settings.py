import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'net'

FEATURES = [
    'feature.SineGen',
    'feature.NoisySineGen',
    'feature.ConvertToClasses',
    'feature.Dropout',
    'feature.Chunk'
]

STAGES = [

]

SAMPLERATE = 11024
DURATION = 1
DROPOUT = 0.50
CLASS_COUNT = 100
FREQUENCY_LIMIT = (300, 800)
LEARNING_RATE = 0.001
DISCRETE_CLASS = 256
BATCH_SIZE = 32
CHUNK = 25


CUDA_VISIBLE_DEVICES = 3

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")
