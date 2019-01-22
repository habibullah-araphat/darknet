from PIL import Image
import os
from evaluate.mAP_utils import read_names


def get_dim(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
    return width, height

def get_annotatoin_dict(input_text, width, height):
    res = dict()
    tokens = input_text.split()
    if len(tokens)!=5:
        return res
    res["class"] = int(tokens[0])
    res["x"] = int((float(tokens[1])-float(tokens[3])/2.0)*width)
    res["y"] = int((float(tokens[2])-float(tokens[4])/2.0)*height)
    res["width"] = int(float(tokens[3])*width)
    res["height"] = int(float(tokens[4])*height)

    return res

def get_annotatoins(img_path):
    base_name = os.path.basename(img_path)
    just_name = os.path.splitext(base_name)[0]
    width, height = get_dim(img_path)
    dir_name = os.path.split(img_path)[0]
    txt_path = os.path.join(dir_name, just_name+".txt")
    annotatoin_list = read_names.get_names(txt_path)
    res_annotation = list()
    for annotatoin in annotatoin_list:
        annotation_dict = get_annotatoin_dict(annotatoin, width, height)
        res_annotation.append(annotation_dict)

    return res_annotation