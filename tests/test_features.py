import numpy as np
import cv2 as cv
from cv_toolkit.features import orb_keypoints, match_orb

def test_orb_and_match():
    img1 = np.zeros((128,128), np.uint8)
    cv.circle(img1, (64,64), 20, 255, -1)
    img2 = cv.GaussianBlur(img1, (3,3), 0.8)
    k1, d1 = orb_keypoints(img1)
    k2, d2 = orb_keypoints(img2)
    assert d1 is not None and d2 is not None
    good = match_orb(d1, d2)
    assert isinstance(good, list)
