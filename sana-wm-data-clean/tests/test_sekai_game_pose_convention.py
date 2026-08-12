import numpy as np
import pytest

from sana_wm_data.ingest.sekai_game import select_sekai_game_c2w


def test_sekai_game_c2w_is_resampled_without_axis_postmultiply():
    raw = np.repeat(np.eye(4, dtype=np.float64)[None], 4, axis=0)
    raw[:, 1, 1] = -1.0
    raw[:, 2, 2] = -1.0
    raw[:, :3, 3] = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    sel = np.array([0, 2, 3], dtype=np.int64)

    selected = select_sekai_game_c2w(raw, sel)

    np.testing.assert_array_equal(selected, raw[sel])
    np.testing.assert_array_equal(selected[:, :3, 3], raw[sel, :3, 3])

    legacy_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    assert not np.array_equal(selected, raw[sel] @ legacy_flip)


def test_sekai_game_c2w_rejects_invalid_inputs():
    with pytest.raises(ValueError, match=r"\[T,4,4\]"):
        select_sekai_game_c2w(np.zeros((3, 3)), np.array([0]))
    with pytest.raises(ValueError, match="integer frame indices"):
        select_sekai_game_c2w(np.zeros((2, 4, 4)), np.array([0.0]))
    with pytest.raises(IndexError, match="outside"):
        select_sekai_game_c2w(np.zeros((2, 4, 4)), np.array([2]))
