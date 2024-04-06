import cv2
import numpy as np
import os


def squareify(img: np.array, side_len: int) -> np.array:
    height, width, _ = img.shape
    square_width = min((width, height))

    square_coords = (
        (width // 2 - square_width // 2, height // 2 - square_width // 2),
        (width // 2 + square_width // 2, height // 2 + square_width // 2)
    )

    frame_mod = img[
        square_coords[0][1]: square_coords[1][1] - 1,
        square_coords[0][0]: square_coords[1][0] - 1
    ]

    downscaled = cv2.resize(frame_mod, (side_len, side_len))

    return downscaled


IMG_SIZE = 64
IMG_READ = "banana"
IMG_SAVE = "scaled"

fruit_dirs = os.listdir(IMG_READ)

print(f"{len(fruit_dirs)} images loaded")

for image in fruit_dirs:
    raw_img = cv2.imread(f"{IMG_READ}/{image}")
    processed = squareify(raw_img, IMG_SIZE)
    cv2.imwrite(f"{IMG_SAVE}/{image}", processed)
