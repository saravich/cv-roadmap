from __future__ import annotations
import cv2 as cv
import numpy as np
from typing import Tuple

def create_tracker(name: str = "KCF"):
    name = name.upper()
    if name == "KCF":
        return cv.TrackerKCF_create()
    if name == "CSRT":
        return cv.TrackerCSRT_create()
    if name == "MIL":
        return cv.TrackerMIL_create()
    raise ValueError(f"Unknown tracker: {name}")

def track_video(path: str, algo: str = "KCF", bbox: Tuple[int,int,int,int] | None = None, out: str | None = None):
    cap = cv.VideoCapture(path)
    ok, frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Cannot open video: {path}")

    if bbox is None:
        # Simple center box
        h, w = frame.shape[:2]
        bbox = (w//4, h//4, w//2, h//2)

    tracker = create_tracker(algo)
    tracker.init(frame, bbox)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(out, fourcc, 30.0, (frame.shape[1], frame.shape[0])) if out else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok, box = tracker.update(frame)
        if ok:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0]+box[2]), int(box[1]+box[3]))
            cv.rectangle(frame, p1, p2, (0,255,0), 2)
        else:
            cv.putText(frame, "Tracking failure", (20,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
        if writer:
            writer.write(frame)
    if writer:
        writer.release()
    cap.release()
