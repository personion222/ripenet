from xml.dom import minidom
from os import listdir
import numpy as np
import qoi

ANN_DIR = "verified/ann"
ANN_SAVE = "raw_orig/ann"
IMG_SAVE = "raw_orig/img"

unripe = []
ripe = []

for ann in listdir(ANN_DIR):
    dom = minidom.parse(f"{ANN_DIR}/{ann}")
    file_path = dom.getElementsByTagName("path")[0].firstChild.nodeValue
    arr_file = qoi.read(file_path, 3)

    is_ripe = int(dom.getElementsByTagName("val")[0].firstChild.nodeValue) == 1

    if is_ripe:
        ripe.append((dom, arr_file, ann))

    else:
        unripe.append((dom, arr_file, ann))

point_len = min((len(unripe), len(ripe)))

np.random.shuffle(ripe)
np.random.shuffle(unripe)

ripe = ripe[:point_len]
unripe = unripe[:point_len]

for img in ripe + unripe:
    jpg_name = img[0].getElementsByTagName("filename")[0].firstChild.nodeValue
    qoi_name = f"{jpg_name.split('.')[0]}.qoi"
    write_dir = f"{IMG_SAVE}/{qoi_name}"
    qoi.write(write_dir, img[1])

    img[0].getElementsByTagName("filename")[0].firstChild.nodeValue = qoi_name
    img[0].getElementsByTagName("path")[0].firstChild.nodeValue = write_dir
    img[0].writexml(open(f"{ANN_SAVE}/{img[2]}", 'w'))
