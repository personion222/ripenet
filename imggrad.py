import cv2
import numpy as np


def activation(x):
    return 1 / (1 + 2.718 ** (-EDGE * (x - 0.5)))


DIR1 = "fathom5.jpg"
DIR2 = "fathom5deutan.jpg"
OUT = "fathom5deutaninterp.jpg"
EDGE = 50

img1 = cv2.resize(cv2.rotate(cv2.imread(DIR1), cv2.ROTATE_90_CLOCKWISE), (0, 0), fx=0.5, fy=0.5)
img2 = cv2.resize(cv2.rotate(cv2.imread(DIR2), cv2.ROTATE_90_CLOCKWISE), (0, 0), fx=0.5, fy=0.5)
imgfade = np.empty(img1.shape, np.uint8)

columns = len(img1)

for colidx, col1 in enumerate(img1):
    col2 = img2[colidx]
    colfade = np.empty(col1.shape, np.uint8)
    for pxidx, px1 in enumerate(col1):
        px2 = col2[pxidx]
        pxfade = np.empty(px1.shape, np.uint8)
        for validx, val1 in enumerate(px1):
            val2 = px2[validx]
            valfade = np.interp(activation(colidx / columns), (0, 1), (val1, val2))
            pxfade[validx] = valfade

        colfade[pxidx] = pxfade

    imgfade[colidx] = colfade

cv2.imwrite(OUT, cv2.rotate(imgfade, cv2.ROTATE_90_COUNTERCLOCKWISE))
