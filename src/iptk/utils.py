from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def imread_gray(path: str | Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imread_color(path: str | Path) -> np.ndarray:
    img = cv.imread(str(path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    if not cv.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write image: {path}")
