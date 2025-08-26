from __future__ import annotations
import cv2 as cv
import numpy as np

def watershed_segmentation(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    _, thr = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, 0.7*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv.watershed(img_bgr, markers)
    seg = np.zeros_like(img_bgr)
    seg[markers > 1] = (0,255,0)
    seg[markers == -1] = (0,0,255)
    return seg
