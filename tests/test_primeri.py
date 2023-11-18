from unittest import TestCase
import numpy as np
import pathlib

from transformacija_sivin import normaliziraj_sivine_uniformno, normaliziraj_sivine_normalno



class TestPrimeri(TestCase):
    #Testi za nalogo 1
    def test_normaliziraj_sivine_uniformno(self):
        # Create a test image with known values
        test_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        # Corrected expected result after normalization
        expected_result = np.array([[0.0, 1/3], [2/3, 1.0]], dtype=np.float32)
        # Perform normalization
        result = normaliziraj_sivine_uniformno(test_image, 0, 1)
        # Check if the result is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


    def test_normaliziraj_sivine_normalno(self):
        # Create a test image with known values
        test_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        mean_of_test_image = test_image.mean()
        std_of_test_image = test_image.std()
        expected_result = (test_image - mean_of_test_image) / std_of_test_image
        # Perform normalization
        result = normaliziraj_sivine_normalno(test_image, 0, 1)
        # Check if the result is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=5)

    def test_uniformno_min_equals_max(self):
        # Test case where the image has the same value for all pixels
        test_image = np.full((2, 2), 100, dtype=np.uint8)
        expected_result = np.full((2, 2), 0, dtype=np.float32)
        result = normaliziraj_sivine_uniformno(test_image, 0, 1)
        np.testing.assert_array_equal(result, expected_result)

    def test_uniformno_invalid_range(self):
        # Test case where vmin is greater than vmax
        test_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            normaliziraj_sivine_uniformno(test_image, 1, 0)

    def test_normalno_zero_std_dev(self):
        # Test case where the standard deviation of the image is zero
        test_image = np.full((2, 2), 100, dtype=np.uint8)
        with self.assertRaises(ZeroDivisionError):
            normaliziraj_sivine_normalno(test_image, 0, 1)

    def test_normalno_invalid_deviation(self):
        # Test case where the desired standard deviation is zero
        test_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            normaliziraj_sivine_normalno(test_image, 0, 0)

