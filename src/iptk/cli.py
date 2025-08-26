from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import typer
from rich import print

from .calibration import calibrate_from_dir
from .features import draw_matches, match_orb, orb_keypoints
from .filters import canny_edges, gaussian_blur
from .geometry import find_homography_ransac
from .optical_flow import lk_flow
from .segmentation import watershed_segmentation
from .tracking import track_video
from .utils import imread_color, imread_gray, imwrite

app = typer.Typer(help="IP Toolkit CLI")


@app.command()
def blur(image: Path, ksize: int = 5, sigma: float = 1.0, out: Path = Path("blur.png")) -> None:
    """Apply Gaussian blur to an image."""
    img = imread_gray(image)
    b = gaussian_blur(img, ksize, sigma)
    imwrite(out, b)
    print(f"[green]Saved[/green] {out}")


@app.command()
def edges(image: Path, low: int = 50, high: int = 150, out: Path = Path("edges.png")) -> None:
    """Detect edges using Canny algorithm."""
    img = imread_gray(image)
    e = canny_edges(img, low, high)
    imwrite(out, e)
    print(f"[green]Saved[/green] {out}")


@app.command()
def match(img1: Path, img2: Path, out: Path = Path("matches.png")) -> None:
    """Find and visualize feature matches between two images."""
    g1, g2 = imread_gray(img1), imread_gray(img2)
    k1, d1 = orb_keypoints(g1)
    k2, d2 = orb_keypoints(g2)
    good = match_orb(d1, d2)
    vis = draw_matches(g1, k1, g2, k2, good)
    imwrite(out, vis)
    print(f"[green]Saved[/green] {out} ({len(good)} matches)")


@app.command()
def homography(img1: Path, img2: Path, out: Path = Path("warped.png")) -> None:
    """Compute homography and warp image."""
    g1, g2 = imread_gray(img1), imread_gray(img2)
    k1, d1 = orb_keypoints(g1)
    k2, d2 = orb_keypoints(g2)
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        typer.echo("Not enough matches")
        raise typer.Exit(code=1)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = find_homography_ransac(pts1, pts2)
    h, w = g1.shape
    warped = cv.warpPerspective(cv.cvtColor(g2, cv.COLOR_GRAY2BGR), H, (w, h))
    imwrite(out, warped)
    print(f"[green]Saved[/green] {out}")


@app.command()
def calibrate(
    folder: Path, pattern: str = "9x6", square: float = 0.024, out: Path = Path("calib.npz")
) -> None:
    """Calibrate camera using chessboard images."""
    res = calibrate_from_dir(folder, pattern, square)
    np.savez(out, **res)
    print(f"[green]Saved[/green] {out} (rms={res['rms']:.4f})")


@app.command()
def lkflow(prev: Path, curr: Path, out: Path = Path("flow.png")) -> None:
    """Compute optical flow using Lucas-Kanade method."""
    p = imread_gray(prev)
    c = imread_gray(curr)
    flow = lk_flow(p, c)
    if flow is None:
        typer.echo("No corners detected")
        raise typer.Exit(code=1)
    p0, p1, st, err = flow
    vis = cv.cvtColor(c, cv.COLOR_GRAY2BGR)
    for (x0, y0), (x1, y1), s in zip(
        p0.reshape(-1, 2), p1.reshape(-1, 2), st.reshape(-1), strict=False
    ):
        if s:
            cv.arrowedLine(
                vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, tipLength=0.3
            )
    imwrite(out, vis)
    print(f"[green]Saved[/green] {out}")


@app.command()
def watershed(image: Path, out: Path = Path("seg.png")) -> None:
    """Apply watershed segmentation to an image."""
    bgr = imread_color(image)
    seg = watershed_segmentation(bgr)
    imwrite(out, seg)
    print(f"[green]Saved[/green] {out}")


@app.command()
def track(video: Path, algo: str = "KCF", out: Path = Path("tracked.mp4")) -> None:
    """Track objects in a video."""
    track_video(str(video), algo=algo, out=str(out))
    print(f"[green]Saved[/green] {out}")


if __name__ == "__main__":
    app()
