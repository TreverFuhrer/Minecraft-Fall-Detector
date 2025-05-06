import pandas as pd
from collections import deque
from features import engineer_features

# Sliding window buffer using fixed size queue
class SlidingBuffer:
    def __init__(self, window_size, model, feature_columns):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.model = model
        self.feature_columns = feature_columns
        self.last_fall_y = None

    def add_tick(self, tick_data):
        self.buffer.append(tick_data)

        # Not enough data to make predition
        if len(self.buffer) < self.window_size:
            return None

        df = pd.DataFrame(list(self.buffer))
        features_df = engineer_features(df.copy())

        # Add live y_diff_from_prev_fall feature
        y_ref = self.last_fall_y if self.last_fall_y is not None else 999
        features_df['y_diff_from_prev_fall'] = features_df['y'] - y_ref
        features_df['has_prev_fall'] = int(self.last_fall_y is not None)

        # Prepare single row for prediction
        X_live = features_df.iloc[[-1]][self.feature_columns]
        fall_prob = self.model.predict_proba(X_live)[0, 1]
        predicted_fall = fall_prob > 0.8
        
        if predicted_fall:
            self.last_fall_y = df.iloc[-1]['y']

        return {
            'fall': predicted_fall,
            'prob': fall_prob,
            'tick_id': tick_data.get('tick_id')
        }
