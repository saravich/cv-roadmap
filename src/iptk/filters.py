from __future__ import annotations

import cv2 as cv
import numpy as np


def gaussian_blur(
    img: np.ndarray[np.uint8, np.dtype[np.uint8]], ksize: int = 5, sigma: float = 1.0
) -> np.ndarray[np.uint8, np.dtype[np.uint8]]:
    ksize = max(3, int(ksize) | 1)  # odd >=3
    return cv.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def canny_edges(
    img: np.ndarray[np.uint8, np.dtype[np.uint8]], low: int = 50, high: int = 150
) -> np.ndarray[np.uint8, np.dtype[np.uint8]]:
    return cv.Canny(img, low, high)
