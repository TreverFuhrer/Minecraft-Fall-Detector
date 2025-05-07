import numpy as np

def engineer_features(df, window=60, last_fall_y=None, last_fall_tick=None, current_tick=0):
    # Movement deltas
    df['deltaY'] = df['y'].diff().fillna(0)
    df['delta_posX'] = df['x'].diff().fillna(0)
    df['delta_posZ'] = df['z'].diff().fillna(0)

    # Direction change detection (within window)
    df['dirX_sign'] = np.sign(df['delta_posX'])
    df['dirZ_sign'] = np.sign(df['delta_posZ'])
    df['dirX_change'] = df['dirX_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    df['dirZ_change'] = df['dirZ_sign'].rolling(window).apply(lambda x: int(np.any(np.diff(x) != 0)), raw=True).fillna(0)
    df['direction_changed'] = ((df['dirX_change'] + df['dirZ_change']) > 0).astype(int)

    # Speed and low-velocity duration
    df['speed'] = np.sqrt(df['velX']**2 + df['velY']**2 + df['velZ']**2)
    df['low_velocity'] = (df['speed'] < 0.03).astype(int)
    df['low_velocity_duration'] = df['low_velocity'].rolling(window).sum().fillna(0)

    # Y-position features
    df['recent_y_min'] = df['y'].rolling(window).min().fillna(df['y'])
    df['y_diff_from_5ago'] = df['y'] - df['y'].shift(5)
    df['y_climb_after_drop'] = ((df['y_diff_from_5ago'] > 1.0) & (df['deltaY'] > 0)).astype(int)

    # Ground state features
    df['onGround'] = df['onGround'].astype(int)
    df['onGround_ratio'] = df['onGround'].rolling(window).mean().fillna(0)

    # Velocity history
    df['velY_prev'] = df['velY'].shift(1)
    df['deltaY_prev'] = df['y'] - df['y'].shift(1)

    # Previous fall tracking (has_prev_fall)
    if 'isFall' in df.columns:
        df['y_prev_fall'] = np.nan
        last_y = None
        for idx in df.index:
            if df.loc[idx, 'isFall'] == 1:
                if last_y is not None:
                    df.loc[idx, 'y_prev_fall'] = last_y
                last_y = df.loc[idx, 'y']
                
        df['y_diff_from_prev_fall'] = df['y'] - df['y_prev_fall']
        df['y_diff_from_prev_fall'] = df['y_diff_from_prev_fall'].fillna(999)
        df['has_prev_fall'] = df['y_prev_fall'].notna().astype(int)

        # Ticks since previous fall (training mode)
        df['ticks_since_prev_fall'] = 999
        last_tick = None
        for idx in df.index:
            if df.loc[idx, 'isFall'] == 1:
                last_tick = idx
            if last_tick is not None:
                df.loc[idx, 'ticks_since_prev_fall'] = idx - last_tick

        df['ticks_since_prev_fall'] = df['ticks_since_prev_fall'].fillna(999)

    else:
        # LIVE mode — use externally tracked last_fall_y and last_fall_tick
        if last_fall_y is not None:
            df['y_diff_from_prev_fall'] = df['y'] - last_fall_y
            df['has_prev_fall'] = 1
        else:
            df['y_diff_from_prev_fall'] = 999
            df['has_prev_fall'] = 0

        if last_fall_tick is not None:
            df['ticks_since_prev_fall'] = current_tick - last_fall_tick
        else:
            df['ticks_since_prev_fall'] = 999

    return df