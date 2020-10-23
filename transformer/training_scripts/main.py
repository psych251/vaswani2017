import transformer
import ml_utils
import torch
import time
import numpy as np

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    ml_utils.training.run_training(transformer.training.train)

