from xml.dom import minidom
from os import listdir

ANN_DIR = "annotated"
IMG_DIR = "scaled"

annotations = listdir(ANN_DIR)

for ann in annotations:
    dom = minidom.parse(f"{ANN_DIR}/{ann}")
    filename = dom.getElementsByTagName("filename")[0].firstChild.nodeValue
    dom.getElementsByTagName("path")[0].firstChild.nodeValue = f"{IMG_DIR}/{filename}"
    dom.writexml(open(f"{ANN_DIR}/{ann}", 'w'))
