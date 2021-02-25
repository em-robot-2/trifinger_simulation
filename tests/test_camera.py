"""Unit tests for the camera module."""
import os
import pytest
import numpy as np
from trifinger_simulation import camera


@pytest.mark.calib_data_to_matrix
def test_calib_data_to_matrix_1x1():
    data = {
        "rows": 1,
        "cols": 1,
        "data": [42],
    }
    expected = np.array([[42]])

    np.testing.assert_array_equal(camera.calib_data_to_matrix(data), expected)


@pytest.mark.calib_data_to_matrix
def test_calib_data_to_matrix_3x3():
    data = {
        "rows": 3,
        "cols": 3,
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    np.testing.assert_array_equal(camera.calib_data_to_matrix(data), expected)


@pytest.mark.calib_data_to_matrix
def test_calib_data_to_matrix_2x4():
    data = {
        "rows": 2,
        "cols": 4,
        "data": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    np.testing.assert_array_equal(camera.calib_data_to_matrix(data), expected)


def test_load_camera_pose_from_calibration_file():
    test_data_dir, _ = os.path.splitext(__file__)
    camera_param_file = os.path.join(test_data_dir, "camera60_full.yml")

    expected_position = np.array([0.2499816, 0.24582271, 0.38930428])
    expected_orientation = np.array(
        [0.36162226, 0.86188589, -0.3288036, -0.13516749]
    )

    position, orientation = camera.load_camera_pose_from_calibration_file(
        camera_param_file
    )

    np.testing.assert_array_almost_equal(position, expected_position)
    np.testing.assert_array_almost_equal(orientation, expected_orientation)
