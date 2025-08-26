# CLI

```bash
cvtk edges input.jpg --out out_edges.png
cvtk blur input.jpg --ksize 5 --sigma 1.0 --out blurred.png
cvtk match img1.jpg img2.jpg --out matches.png
cvtk homography img1.jpg img2.jpg --out warped.png
cvtk calibrate chessboard_dir --pattern 9x6 --square 0.024  # meters
cvtk lkflow frame1.jpg frame2.jpg --out flow.png
cvtk watershed input.jpg --out seg.png
cvtk track video.mp4 --algo KCF --out tracked.mp4
```
