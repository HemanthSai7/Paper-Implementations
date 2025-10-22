from dataclasses import dataclass

@dataclass
class Config:
    N_EMBED = 64
    BATCH_SIZE = 64
    BLOCK_SIZE = 64
    NUM_HEADS = 4
    NUM_BLOCKS = 4
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 100000
    EVAL_INTERVAL = 250
    EVAL_STEPS = 100
    GENERATION_TOKENS = 200
