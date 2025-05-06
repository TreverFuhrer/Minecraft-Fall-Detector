from core.buffer import Buffer

def test_buffer_keeps_window():
    buffer = Buffer(window_size=3)

    # Simulate incoming ticks
    buffer.add_tick({'tick_id': 1, 'y': 100})
    buffer.add_tick({'tick_id': 2, 'y': 99})
    buffer.add_tick({'tick_id': 3, 'y': 98})
    buffer.add_tick({'tick_id': 4, 'y': 97})  # Should remove tick_id 1

    df = buffer.get_buffer_df()
    assert len(df) == 3
    assert df.iloc[0]['tick_id'] == 2
