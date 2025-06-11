import torch
from src.utils import get_device

DEVICE = get_device()

PRETRAINED_MODEL_NAME = 'bert-base-uncased'

ENCODER_HIDDEN_DIM = 128
DECODER_HIDDEN_DIM = 128
ATTENTION_DIM = 64

LEARNING_RATE = 0.0005
BATCH_SIZE = 8
NUM_EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD_NORM = 1.0

RAW_DATA_PATH = 'data/raw_dataset.txt'
MAX_SRC_SEQ_LENGTH = 100
MAX_TRG_SEQ_LENGTH = 75

MODEL_SAVE_DIR = 'models_checkpoint'
MODEL_SAVE_PATH = f'{MODEL_SAVE_DIR}/qa_chatbot_gru.pt'
CONFIG_SAVE_PATH = f'{MODEL_SAVE_DIR}/config.json'
LOG_INTERVAL = 10

import os
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
