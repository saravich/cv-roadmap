from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def _grid_from_str(pattern: str) -> tuple[int, int]:
    cols, rows = pattern.lower().split("x")
    return int(cols), int(rows)


def calibrate_from_dir(
    folder: str | Path,
    pattern: str = "9x6",
    square_size: float = 1.0,
) -> dict[str, float | np.ndarray | list[np.ndarray]]:
    folder = Path(folder)
    cols, rows = _grid_from_str(pattern)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints, imgpoints = [], []
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")

    img_size = None
    for p in images:
        img = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
        ret, corners = cv.findChessboardCorners(img, (cols, rows))
        if ret:
            corners2 = cv.cornerSubPix(
                img,
                corners,
                (11, 11),
                (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_size = img.shape[::-1]

    if not objpoints:
        raise RuntimeError("No chessboard detections; check pattern and images.")

    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return {"rms": float(ret), "K": K, "dist": dist, "rvecs": rvecs, "tvecs": tvecs}
