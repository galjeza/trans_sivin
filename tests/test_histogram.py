from unittest import TestCase
import numpy as np
from transformacija_sivin import izenaci_histogram

class TestIzenaciHistogram(TestCase):
    
    def test_histogram_equalization_full_range(self):
        # Uporaba deterministične slike z gradientom, ki pokriva celoten obseg
        test_image = np.linspace(0, 255, 256, endpoint=True).astype(np.float32).reshape((16, 16))
        equalized_image = izenaci_histogram(test_image, vmin=0, vmax=255)
        # Preverjanje, ali so maksimalne in minimalne vrednosti izenačene slike blizu vmax in vmin
        # Sproščena toleranca zaradi morebitnih napak na robovih
        self.assertTrue(np.isclose(equalized_image.max(), 255, atol=5))
        self.assertTrue(np.isclose(equalized_image.min(), 0, atol=5))
    
    def test_histogram_equalization_narrow_range(self):
        # Test izenačevanja histograma v ozkem obsegu, ki ne sme preseči tega obsega
        test_image = np.random.rand(10, 10).astype(np.float32) * 0.5 + 0.25
        equalized_image = izenaci_histogram(test_image, vmin=0.25, vmax=0.75)
        # Preverjanje, ali so vrednosti izenačene slike znotraj obsega
        self.assertTrue(np.all(equalized_image >= 0.25) and np.all(equalized_image <= 0.75))

    def test_histogram_equalization_small_bins(self):
        # Uporaba večje, bolj enakomerno razporejene testne slike
        test_image = np.tile(np.linspace(0, 1, 10), (10, 1))
        equalized_image = izenaci_histogram(test_image, vmin=0, vmax=1, bins=10)
        hist, _ = np.histogram(equalized_image, bins=10, range=(0, 1))
        # Preverjanje, ali v histogramu ni praznih intervalov
        self.assertTrue(np.all(hist > 0), f"Prazni intervali najdeni: {hist}")

    def test_histogram_equalization_large_bins(self):
        # Test izenačevanja histograma z velikim številom intervalov
        test_image = np.random.rand(10, 10).astype(np.float32)
        equalized_image = izenaci_histogram(test_image, vmin=0, vmax=1, bins=1024)
        hist, bins = np.histogram(equalized_image, bins=1024, range=(0, 1))
        cdf = np.cumsum(hist).astype(float)
        cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())
        # Preverjanje, ali je kumulativna porazdelitvena funkcija blizu linearni
        ideal_cdf = np.linspace(0, 1, 1024)
        self.assertTrue(np.allclose(cdf_normalized, ideal_cdf, atol=0.1))

    def test_histogram_equalization_interpolation_edge_case(self):
        # Test izenačevanja histograma z interpolacijo na robu obsega
        test_image = np.array([0, 0.5, 1]).astype(np.float32)
        equalized_image = izenaci_histogram(test_image, vmin=0, vmax=1, interpolacija=True)
        # Interpolacija ne sme potisniti vrednosti preko izvirnega obsega
        self.assertTrue(np.all(equalized_image >= 0) and np.all(equalized_image <= 1))
