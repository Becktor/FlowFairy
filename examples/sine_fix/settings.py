import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'net'

FEATURES = [
    'feature.FrequencyGen',
    'feature.SineGen',
    'feature.NoisySineGen',
    'feature.Dropout',
    'feature.Mask'
]

STAGES = [
    
]

SAMPLERATE = 11024
DURATION = 1
DROPOUT = 0.50
LEARNING_RATE = 0.005

BATCH_SIZE = 32

CUDA_VISIBLE_DEVICES = 0

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")
