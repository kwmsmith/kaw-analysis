import numpy as np
import pdfs


def test_fit_gaussian():
    data = 10*np.random.normal(loc=1.0, scale=2.0, size=(100,100))
    center, std_dev = pdfs.fit_gaussian(data)
    assert abs(center - 10) < 0.5
    assert abs(std_dev - 20) < 0.5
