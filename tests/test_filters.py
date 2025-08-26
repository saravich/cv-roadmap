import numpy as np

from iptk.filters import canny_edges, gaussian_blur


def test_edges_shape() -> None:
    img = np.zeros((64, 64), np.uint8)
    img[16:48, 16:48] = 255
    edges = canny_edges(img, 50, 150)
    assert edges.shape == img.shape
    assert edges.dtype == np.uint8


def test_blur_basic() -> None:
    img = np.zeros((32, 32), np.uint8)
    img[16, 16] = 255
    b = gaussian_blur(img, 5, 1.0)
    assert b.shape == img.shape
    assert b.dtype == np.uint8
