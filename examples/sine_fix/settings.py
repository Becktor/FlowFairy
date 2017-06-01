import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'netops'

FEATURES = [
    #'feature.SineGen',
    #'feature.NoisySineGen',
    #'feature.Dropout',
    'feature.Speech',
    'feature.ConvertToClasses'
    #'feature.Mask'
]

SAMPLERATE = 22050
OUTPUTLEN = 11024
DURATION = 1
DROPOUT = 0.50
LEARNING_RATE = 0.001 # 5e-4
CLASS_COUNT = 2483
FREQUENCY_START = 200
FREQUENCY_LIMIT = (FREQUENCY_START, FREQUENCY_START + CLASS_COUNT*2)
EMBEDDING_SIZE = 2
DISCRETE_CLASS = 256
MAX_AMP = 3


BATCH_SIZE = 32
QUEUE_CAPACITY = 4 * BATCH_SIZE

CUDA_VISIBLE_DEVICES = 2

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")

DATA = {
    'train': (),
    'validation': (),
    'test': ()
}
