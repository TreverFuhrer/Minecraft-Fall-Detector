import pandas as pd
from collections import deque
from .features import engineer_features

class SlidingBuffer:
    def __init__(self, window_size, model, features):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.model = model
        self.features = features
        self.last_fall_y = None
        self.last_fall_tick = None  # Track the last fall tick for live mode
        self.current_tick = 0  # Incremented on each new tick

    def add_tick(self, tick_data):
        self.buffer.append(tick_data)
        self.current_tick += 1

        # Not enough data to make a prediction
        if len(self.buffer) < self.window_size:
            return None

        # Generate features from the current buffer
        df = pd.DataFrame(list(self.buffer))

        # Use the new engineer_features function
        features_df = engineer_features(
            df.copy(),
            last_fall_y=self.last_fall_y,
            last_fall_tick=self.last_fall_tick,
            current_tick=self.current_tick
        )

        # Select the newest row of features
        X_live = features_df.iloc[[-1]][self.features]
        fall_prob = self.model.predict_proba(X_live)[0, 1]
        predicted_fall = fall_prob > 0.8

        # Update the last fall state if a fall is predicted
        if predicted_fall:
            self.last_fall_y = df.iloc[-1]['y']
            self.last_fall_tick = self.current_tick  # Reset tick counter

        return {
            'fall': predicted_fall,
            'prob': fall_prob,
            'tick_id': tick_data.get('tick_id')
        }
