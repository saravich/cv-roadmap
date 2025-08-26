from __future__ import annotations
import cv2 as cv
import numpy as np

def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    ksize = max(3, int(ksize) | 1)  # odd >=3
    return cv.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def canny_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv.Canny(img, low, high)
