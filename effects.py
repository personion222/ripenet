from scipy import ndimage
from PIL import Image
import numpy as np


def rotate_img(image, angle, bgcol):
    rotated = ndimage.rotate(image, angle, cval=127, reshape=False)

    bg = np.where(rotated == 127)
    rotated[bg[0], bg[1], :] = bgcol

    return rotated


def shift_img(image, shift, bgcol):
    pil_img = Image.fromarray(image)
    shifted = Image.new("RGB", image.shape[: 2], tuple(bgcol))

    shifted.paste(pil_img, tuple(shift))

    return np.asarray(shifted, dtype=np.uint8)


def zoom_img(image, scale, bgcol):
    c = image.shape[0] // 2, image.shape[1] // 2
    scaled = ndimage.zoom(image, (scale, scale, 1))

    zoomed = np.empty(image.shape, dtype=np.uint8)
    zoomed[:] = bgcol

    if scale < 1:
        zoomed[c[0] - scaled.shape[0] // 2: c[0] - scaled.shape[0] // 2 + scaled.shape[0],
               c[1] - scaled.shape[1] // 2: c[1] - scaled.shape[1] // 2 + scaled.shape[1]] = scaled

        return zoomed

    zoomed = scaled[scaled.shape[0] // 2 - c[0]: scaled.shape[0] // 2 + c[0],
                    scaled.shape[1] // 2 - c[1]: scaled.shape[1] // 2 + c[1]]

    return zoomed


def add_noise(img, fac):
    img16 = img.astype(np.int16)
    noise = ((np.random.randn(img.shape[0], img.shape[1], img.shape[2]) - 0.5) * fac).astype(np.int16)
    img16 += noise
    np.clip(img16, 0, 255, out=img16)
    img_out = img16.astype(np.uint8)

    return img_out
