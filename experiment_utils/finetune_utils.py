import time
def current_milli_time():
    return int(round(time.time() * 1000))
class FineTuneConfig:
  def __init__(self) -> None:
      self.MAX_SEQ_LEN = 256
      self.TRAIN_BATCH_SIZE = 16
      self.VALID_BATCH_SIZE = 8
      self.EPOCHS = 4
      self.SPLITS = 4
      self.LEARNING_RATE = 5e-5
      self.WARMUP_RATIO = 0.1
      self.MAX_GRAD_NORM = 1.0
      self.ACCUMULATION_STEPS = 1
