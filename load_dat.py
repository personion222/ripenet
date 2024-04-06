import qoi
from xml.dom import minidom
from os import listdir
import numpy as np
from scipy import ndimage
from random import sample
from time import time


class DataPoint:
    def __init__(self, img_dat: np.ndarray, cont_box, fruit: str, ripeness: int, filename: str):
        self.filename = filename
        self.dims = img_dat.shape
        self.cont_dims = (48, 48, 3)
        self.img = img_dat
        self.cont_box = cont_box
        self.type = np.array((int(fruit == "apple"), int(fruit == "banana")), np.uint8)
        self.ripeness = ripeness
        self.cropped = img_dat[int(cont_box[1] * 64): int((cont_box[1] + cont_box[3]) * 64),
                               int(cont_box[0] * 64): int((cont_box[0] + cont_box[2]) * 64)]

        '''print(int(cont_box[1] * 64), int((cont_box[1] + cont_box[3]) * 64),
              int(cont_box[0] * 64), int((cont_box[0] + cont_box[2]) * 64))'''

        self.content = ndimage.zoom(self.cropped, (48 / self.cropped.shape[0], 48 / self.cropped.shape[1], 1))


def load_dat(dat_dir: str, freq: float = 1):
    annotation_files = listdir(dat_dir)
    annotation_files = sample(annotation_files, int(len(annotation_files) * freq))
    inputs = list()
    times = {
        "dom_parse": [],
        "img_load": [],
        "img_parse": [],
        "dom_read": [],
        "total": []
    }

    for ann_idx, ann in enumerate(annotation_files):
        full = time()
        st = time()
        parsed = minidom.parse(f"{dat_dir}/{ann}")
        times["dom_parse"].append(time() - st)

        st = time()
        img_src = parsed.getElementsByTagName("path")[0].firstChild.nodeValue
        img_arr = qoi.read(img_src, 3)
        times["img_load"].append(time() - st)

        st = time()
        img_arr = np.divide(img_arr, 255)
        img_size = int(parsed.getElementsByTagName("width")[0].firstChild.nodeValue)
        times["img_parse"].append(time() - st)

        st = time()
        try:
            xmin = int(parsed.getElementsByTagName("xmin")[0].firstChild.nodeValue)
            ymin = int(parsed.getElementsByTagName("ymin")[0].firstChild.nodeValue)
            xmax = int(parsed.getElementsByTagName("xmax")[0].firstChild.nodeValue)
            ymax = int(parsed.getElementsByTagName("ymax")[0].firstChild.nodeValue)

        except IndexError:
            print(ann)
            quit()

        bnd_box = np.array((xmin / img_size, ymin / img_size, (xmax - xmin) / img_size, (ymax - ymin) / img_size), np.half)

        fruit_type = parsed.getElementsByTagName("name")[0].firstChild.nodeValue

        ripeness = int(parsed.getElementsByTagName("val")[0].firstChild.nodeValue)

        times["dom_read"].append(time() - st)

        inputs.append(DataPoint(img_arr, bnd_box, fruit_type, ripeness, ann))
        times["total"].append(time() - full)

    return np.asarray(inputs), times
