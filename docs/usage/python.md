# Python API

```python
import cv2 as cv
import numpy as np
from cv_toolkit.filters import canny_edges

img = cv.imread("input.jpg", cv.IMREAD_GRAYSCALE)
edges = canny_edges(img, 50, 150)
```

See module pages in the sidebar for more.
