import pandas as pd
import numpy as np
import joblib
from core.features import engineer_features
from core.buffer import SlidingBuffer

# Watchdog detects new tick
# Sends new tick to sliding buffer
# Buffer queues new tick, max queue size n

# === MAIN TRAINING FLOW ===
def main():
    # Load model
    model = joblib.load('fall_model.joblib')

    # Rrolling buffer of 10 ticks
    buffer = SlidingBuffer(10, model)

    # watchdog reads new tick and adds it to buffer