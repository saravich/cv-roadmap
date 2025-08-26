from __future__ import annotations

import cv2 as cv
import numpy as np


def find_homography_ransac(
    pts1: np.ndarray[np.float32, np.dtype[np.float32]],
    pts2: np.ndarray[np.float32, np.dtype[np.float32]],
    ransac_reproj_thr: float = 3.0,
) -> tuple[np.ndarray[np.float64, np.dtype[np.float64]], np.ndarray[np.uint8, np.dtype[np.uint8]]]:
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=ransac_reproj_thr)
    return H, mask
