from os import listdir, system, get_terminal_size
from xml.dom import minidom
import numpy as np
import qoi
import cv2


def color_text(txt, bg=None, fg=None):
    if bool(bg) != bool(fg):
        if fg:
            color_txt = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m{txt}\033[0m"

        else:
            color_txt = f"\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{txt}\033[0m"

    elif bg is None:
        color_txt = txt

    else:
        color_txt = f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{txt}\033[0m"

    return color_txt


def disp_arr(arr, sidetxt='', overtxt=''):
    s_txtlines = sidetxt.split('\n')
    s_linecount = len(s_txtlines)
    assert len(arr) >= s_linecount

    o_txtlines = overtxt.split('\n')
    o_linecount = len(o_txtlines)
    assert len(arr) >= o_linecount

    for idx, row in enumerate(arr):
        for pxidx, px in enumerate(row):
            fg = None
            if idx < o_linecount and pxidx < len(o_txtlines[idx]):
                char = o_txtlines[idx][pxidx]
                if np.average(px) > 127:
                    col = 0
                else:
                    col = 255
                fg = tuple(np.full(3, col, np.uint8))

            else:
                char = ' '

            print(color_text(char, bg=tuple(px), fg=fg), end='')

        if idx < s_linecount:
            print('\t' + s_txtlines[idx])

        else:
            print()


def get_tag(dom, tag):
    return dom.getElementsByTagName(tag)[0].firstChild.nodeValue


def put_tag(dom, tag, val):
    dom.getElementsByTagName(tag)[0].firstChild.nodeValue = val


ANN_DIR = "tmp/ann"
VER_IMG = "verified/img"
VER_ANN = "verified/ann"
JPG = False

anns = listdir(ANN_DIR)
imgs = list()

for ann in anns:
    ann_dom = minidom.parse(f"{ANN_DIR}/{ann}")

    img_src = get_tag(ann_dom, "path")

    if JPG:
        img_arr = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2RGB)

    else:
        img_arr = qoi.read(img_src, 3)

    imgs.append((ann_dom, img_arr))

w = get_terminal_size().columns
h = get_terminal_size().lines

verified = [True] * len(anns)
img_size = ((h - 5) * 2, h - 5)

system("clear")

img_ptr = 0

while True:
    img = imgs[img_ptr]
    ripe = get_tag(img[0], "val") == '1'

    ver = verified[img_ptr]

    if ripe:
        ripetxt = "ripe"
    else:
        ripetxt = "unripe"

    if ver:
        vertxt = "verified"
    else:
        vertxt = "unverified"

    imgpath = "./" + get_tag(img[0], "path")
    prog = img_ptr / (len(imgs) - 1)

    disp_arr(cv2.resize(img[1], img_size), overtxt=f"\n {imgpath}\n {ripetxt}\n {vertxt}", sidetxt='\n' + img[0].toxml())

    print(f"\n{str(round(prog * 100)):>3}%: {color_text(round((img_size[0] - 6) * prog) * ' ', bg=(120, 120, 255))}", end='')
    print(color_text(round((img_size[0] - 6) * (1 - prog)) * ' ', bg=(80, 80, 80)))

    command = input(color_text(f"\ncmd> ", fg=())).lower()

    if command == 'd':
        if img_ptr < len(imgs) - 1:
            img_ptr += 1
        else:
            print('\a')

    elif command == 'a':
        if img_ptr > 0:
            img_ptr -= 1
        else:
            print('\a')

    elif command == 'w':
        verified[img_ptr] = not verified[img_ptr]
        if not verified[img_ptr]:
            if img_ptr < len(imgs) - 1:
                img_ptr += 1
            else:
                print('\a')

    elif command == 's':
        if ripe:
            put_tag(img[0], "val", '0')
        else:
            put_tag(img[0], "val", '1')

    elif command.split(' ')[0] == "tp":
        try:
            tpval = int(command.split(' ')[1])
            assert tpval <= 100
            img_ptr = round((tpval / 100 * (len(imgs) - 1)))

        except (IndexError, TypeError, AssertionError):
            print('\a')

    elif command == "exit":
        break

    else:
        print('\a')

    system("clear")

for idx, isver in enumerate(verified):
    if isver:
        img = imgs[idx]
        filename = get_tag(img[0], "filename").split('.')[0]
        qoi_name = f"{filename}.qoi"
        qoi_path = f"{VER_IMG}/{qoi_name}"
        qoi.write(qoi_path, img[1])

        put_tag(img[0], "filename", qoi_name)
        put_tag(img[0], "path", qoi_path)
        put_tag(img[0], "path", qoi_path)

        img[0].writexml(open(f"{VER_ANN}/{filename}.xml", 'w'))
