from math import sin, cos, radians
from multiprocessing import Pool
from os.path import basename
from xml.dom import minidom
from os import listdir
import numpy as np
import effects
import random
import time
import qoi


def rot_point(poi, ori, angle):
    rads = radians(angle)
    rel_poi = (poi[0] - ori[0], poi[1] - ori[1])
    rotated = (rel_poi[0] * cos(rads) - rel_poi[1] * sin(rads),
               -(rel_poi[1] * cos(rads) + rel_poi[0] * sin(rads)))

    out = np.round((rotated[0] + ori[0], rotated[1] + ori[1]))

    return out


def scale_point(poi, ori, scale):
    rel_poi = (poi[0] - ori[0], poi[1] - ori[1])
    scaled = (rel_poi[0] * scale,
              rel_poi[1] * scale)

    out = np.round((scaled[0] + ori[0], scaled[1] + ori[1]))

    return out


def process_ann(ann):
    ann_name = ann.split('.')

    dom = minidom.parse(f"{ANN_DIR}/{ann}")
    file_path = dom.getElementsByTagName("path")[0].firstChild.nodeValue

    img_dims = dom.getElementsByTagName("size")[0]
    img_width = int(img_dims.getElementsByTagName("width")[0].firstChild.nodeValue)
    img_height = int(img_dims.getElementsByTagName("height")[0].firstChild.nodeValue)
    c = (img_width // 2, img_height // 2)

    bndbox = dom.getElementsByTagName("bndbox")[0]
    xmin = int(bndbox.getElementsByTagName("xmin")[0].firstChild.nodeValue)
    ymin = int(bndbox.getElementsByTagName("ymin")[0].firstChild.nodeValue)
    xmax = int(bndbox.getElementsByTagName("xmax")[0].firstChild.nodeValue)
    ymax = int(bndbox.getElementsByTagName("ymax")[0].firstChild.nodeValue)

    box_points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))

    img_name = basename(file_path).split('.')
    img_raw = qoi.read(file_path, 3)
    qoi.write(f"{ANN_IMG}/{basename(file_path)}", img_raw)
    dom.getElementsByTagName("path")[0].firstChild.nodeValue = f"{ANN_IMG}/{basename(file_path)}"

    for i in range(ADULT_VAL):
        img_name_new = f"{img_name[0]}-{i}.{img_name[1]}"
        img_dir = f"{ANN_IMG}/{img_name_new}"
        if i > 0:
            angle = random.randint(0, 359)
            bgcol = np.random.randint(0, 255, size=3)
            scale = np.random.uniform(0.8, 1.3)
            shift = np.random.randint(-12, 12, size=2)

            mod_img = effects.rotate_img(img_raw, angle, bgcol)
            mod_img = effects.zoom_img(mod_img, scale, bgcol)
            mod_img = effects.shift_img(mod_img, shift, bgcol)

            noise_fac = random.randint(0, 16)
            mod_img = effects.add_noise(mod_img, noise_fac)

            box_points_rot = np.empty(box_points.shape, dtype=np.uint8)

            for idx, point in enumerate(box_points):
                box_points_rot[idx] = rot_point(point, c, angle)

            xmin_proc, ymin_proc = np.min(box_points_rot, axis=0)
            xmax_proc, ymax_proc = np.max(box_points_rot, axis=0)

            xmin_proc, ymin_proc = scale_point((xmin_proc, ymin_proc), c, scale)
            xmax_proc, ymax_proc = scale_point((xmax_proc, ymax_proc), c, scale)

            xmin_proc += shift[0]
            xmax_proc += shift[0]
            ymin_proc += shift[1]
            ymax_proc += shift[1]

            xmin_proc = round(np.clip(xmin_proc, 0, img_width))
            ymin_proc = round(np.clip(ymin_proc, 0, img_height))
            xmax_proc = round(np.clip(xmax_proc, 0, img_width))
            ymax_proc = round(np.clip(ymax_proc, 0, img_height))

            if xmin_proc >= xmax_proc or ymin_proc >= ymax_proc:
                continue

            bndbox.getElementsByTagName("xmin")[0].firstChild.nodeValue = str(xmin_proc)
            bndbox.getElementsByTagName("ymin")[0].firstChild.nodeValue = str(ymin_proc)
            bndbox.getElementsByTagName("xmax")[0].firstChild.nodeValue = str(xmax_proc)
            bndbox.getElementsByTagName("ymax")[0].firstChild.nodeValue = str(ymax_proc)

            qoi.write(img_dir, mod_img)

            dom.getElementsByTagName("filename")[0].firstChild.nodeValue = img_name_new
            dom.getElementsByTagName("path")[0].firstChild.nodeValue = img_dir

        dom.writexml(open(f"{ANN_SAVE}/{ann_name[0]}-{i}.{ann_name[1]}", 'w'))

    dom.unlink()


NOISE_MAX = 36
ADULT_VAL = 15

ANN_DIR = "raw_orig/ann"
ANN_SAVE = "fab_qoi/ann"
ANN_IMG = "fab_qoi/img"

annotations = listdir(ANN_DIR)

start = time.time()
with Pool() as p:
    p.map(process_ann, annotations)

print(f"finished in {round(time.time() - start, 2)} secs")
