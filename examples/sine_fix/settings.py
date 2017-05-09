import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'net_classify'

FEATURES = [
    'feature.SineGen',
    'feature.NoisySineGen',
    'feature.Dropout',
    'feature.ConvertToClasses'
    #'feature.Mask'
]

SAMPLERATE = 11024
DURATION = 1
DROPOUT = 0.50
LEARNING_RATE = 0.001
CLASS_COUNT = 200
FREQUENCY_LIMIT = (120, 720)
EMBEDDING_SIZE = 1
EMBEDDING_INPUT_SIZE = 1
DISCRETE_CLASS = 256


BATCH_SIZE = 64
QUEUE_CAPACITY = 64 * BATCH_SIZE

CUDA_VISIBLE_DEVICES = 0

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")

DATA = {
    'train': (),
    'validation': (),
    'test': ()
}
