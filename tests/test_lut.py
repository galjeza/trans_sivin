from unittest import TestCase
import numpy as np
import pathlib

from transformacija_sivin import transformiraj_z_lut

class TestTransformirajZLUT(TestCase):
    
    def test_basic_transformation(self):
        # Test basic transformation with known LUT
        image = np.array([[0, 1], [2, 3]], dtype=np.uint16)
        lut = np.array([10, 20, 30, 40], dtype=int)
        expected_transformed_image = np.array([[10, 20], [30, 40]], dtype=int)
        result = transformiraj_z_lut(image, lut)
        np.testing.assert_array_equal(result, expected_transformed_image)
    
    def test_clipping_high_values(self):
        # Test transformation where image values exceed the LUT max index, should clip to the last LUT value
        image = np.array([[4, 5], [6, 7]], dtype=np.uint16)
        lut = np.array([10, 20, 30, 40], dtype=int)
        expected_transformed_image = np.array([[40, 40], [40, 40]], dtype=int)
        result = transformiraj_z_lut(image, lut)
        np.testing.assert_array_equal(result, expected_transformed_image)
    
    def test_clipping_low_values(self):
        # Test transformation where image values are zero, should use the first LUT value
        image = np.array([[0, 0], [0, 0]], dtype=np.uint16)
        lut = np.array([10, 20, 30, 40], dtype=int)
        expected_transformed_image = np.array([[10, 10], [10, 10]], dtype=int)
        result = transformiraj_z_lut(image, lut)
        np.testing.assert_array_equal(result, expected_transformed_image)