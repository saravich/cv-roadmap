from __future__ import annotations

import cv2 as cv
import numpy as np


def find_homography_ransac(
    pts1: np.ndarray, pts2: np.ndarray, ransac_reproj_thr: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, ransacReprojThreshold=ransac_reproj_thr)
    return H, mask
