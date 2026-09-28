import numpy as np

from app.hand_track import motion_valley_hand_indices


def test_motion_valley_finds_low_energy_gap():
    motion = np.array([0.9, 1.0, 0.95, 0.1, 0.08, 0.12, 0.95, 1.0, 0.9], dtype=np.float32)
    valleys = motion_valley_hand_indices(motion, smooth_radius=1, valley_ratio=0.5, minimum_gap=2)
    assert valleys
    assert 2 <= valleys[0] <= 5
