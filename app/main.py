import joblib
from core.buffer import SlidingBuffer
import time

# === Parse incoming CSV line to tick dictionary ===
def parse_tick_line(line):
    # Assumes header: tick,x,y,z,velX,velY,velZ,onGround,isFall
    parts = line.strip().split(',')

    # Skip malformed lines
    if len(parts) != 9:
        return None

    # Features dictionary
    return {
        'tick_id': int(parts[0]),
        'x': float(parts[1]),
        'y': float(parts[2]),
        'z': float(parts[3]),
        'velX': float(parts[4]),
        'velY': float(parts[5]),
        'velZ': float(parts[6]),
        'onGround': parts[7].strip().lower() == 'true',
        # 'isFall' is ignored in real-time use
    }

# === Tail a CSV file for new tick lines ===
def tail_csv(file_path, callback, sleep_interval=0.01):
    with open(file_path, 'r') as f:
        f.seek(0, 2)  # Go to end of file

        while True:
            line = f.readline()
            if not line:
                time.sleep(sleep_interval)
                continue

            tick_data = parse_tick_line(line)
            print("new line: " + str(tick_data))
            if tick_data:
                callback(tick_data)


# === Main Inference Runner ===
def main():
    
    # Load trained fall detection model
    model = joblib.load('models/fall_model5.joblib')
    
    # Features used by model
    features = [
        'onGround',
        'onGround_ratio',
        'y',
        'recent_y_min',
        'deltaY',
        'low_velocity_duration',
        'direction_changed',
        'velY_prev',
        #'y_diff_from_prev_fall',
        'delta_posX',
        'delta_posZ',
        'dirX_sign',
        'dirZ_sign',
        'dirX_change',
        'dirZ_change',
        'speed',
        'low_velocity',
        'y_diff_from_5ago',
        'y_climb_after_drop',
        'deltaY_prev',
        #'y_prev_fall',
        'has_prev_fall',
        'ticks_since_prev_fall'
    ]
    
    # Initialize sliding buffer for real-time inference
    buffer = SlidingBuffer(window_size=60, model=model, features=features)
    
    def handle_tick(tick_data):
        result = buffer.add_tick(tick_data)
        if result and result['fall']:
            print("==========\n")
            print(f"[FALL] tick={result['tick_id']} | prob={result['prob']:.3f}")
            print("==========\n")

    # Start monitoring tick stream
    tick_file = 'data/test.csv'  # path to your live tick file
    print("-- loaded test.csv --")
    tail_csv(tick_file, handle_tick)


if __name__ == '__main__':
    main()