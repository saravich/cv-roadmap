from __future__ import annotations

import cv2 as cv
import numpy as np


def lk_flow(
    prev: np.ndarray[np.uint8, np.dtype[np.uint8]],
    curr: np.ndarray[np.uint8, np.dtype[np.uint8]],
    max_corners: int = 200,
) -> (
    tuple[
        np.ndarray[np.float32, np.dtype[np.float32]],
        np.ndarray[np.float32, np.dtype[np.float32]],
        np.ndarray[np.uint8, np.dtype[np.uint8]],
        np.ndarray[np.float32, np.dtype[np.float32]],
    ]
    | None
):
    p0 = cv.goodFeaturesToTrack(prev, maxCorners=max_corners, qualityLevel=0.01, minDistance=7)
    if p0 is None:
        return None
    p1, st, err = cv.calcOpticalFlowPyrLK(prev, curr, p0, None)
    return p0, p1, st, err
