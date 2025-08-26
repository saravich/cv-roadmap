from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Tuple, List

def orb_keypoints(img: np.ndarray, nfeatures: int = 1000):
    orb = cv.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(img, None)
    return kps, desc

def match_orb(desc1: np.ndarray, desc2: np.ndarray, max_ratio: float = 0.75) -> List[cv.DMatch]:
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < max_ratio * n.distance]
    return good

def draw_matches(img1, kps1, img2, kps2, matches):
    return cv.drawMatches(img1, kps1, img2, kps2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
