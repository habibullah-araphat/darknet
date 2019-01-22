from evaluate.mAP_utils import load_config, yolo_annotations, ap_calculator, image_annotations, read_names
import sys
import os

profile = "DEFAULT"
class CalculateMAP:
    def init_pars(self, config):
        self.darknet_dir = config[profile]["darknet_dir"]
        self.obj_path = config[profile]["obj_path"]
        self.names_path = config[profile]["names_path"]
        self.cfg_path = config[profile]["cfg_path"]
        self.weight_path = config[profile]["weight_path"]
        self.image_list_path = config[profile]["image_list_path"]
        self.min_iou = float(config[profile]["min_iou"])

    def get_actual_dict(self, image_list):
        actual_dict = dict()
        for image in image_list:
            actual_dict[image] = image_annotations.get_annotatoins(image)
        return actual_dict

    def get_prediction_dict(self, image_list, names_dict):
        prediction_dict = dict()
        total_image_num = len(image_list)
        image_number = 0
        print("total images: %6d" % (total_image_num))

        for image in image_list:
            image_number += 1
            #print(image_number, image)
            progress = (image_number/total_image_num)*100.00
            sys.stdout.write("\r%8.2f%%%5d: %s" % (progress, image_number, image))
            prediction_dict[image] = yolo_annotations.get_annotation(self.darknet_dir, self.obj_path, self.cfg_path, self.weight_path, image)
            for idx in range(len(prediction_dict[image])):
                prediction_dict[image][idx]["class"] = names_dict[prediction_dict[image][idx]["class"]]

            sys.stdout.flush()
        return prediction_dict

    def evaluatex(self, config_path):
        config = load_config.get_config(config_path)
        self.init_pars(config)
        names_list = read_names.get_names(self.names_path)
        names_dict = read_names.get_name_map(names_list)
        image_list = read_names.get_names(self.image_list_path)

        actual_dict = self.get_actual_dict(image_list)
        prediction_dict = self.get_prediction_dict(image_list, names_dict)
        totoal_ap = 0.0
        mAP_score = 0.0
        print("\nAP:")
        for idx in range(len(names_list)):
            cur_ap = ap_calculator.calculate_ap(actual_dict, prediction_dict, idx, self.min_iou)
            totoal_ap += cur_ap
            print("class:%sap:%8.2f%%" % (names_list[idx], cur_ap*100.00))

        if len(names_list)>0:
            mAP_score = totoal_ap/float(len(names_list))
        print("mAP@%.2f:%8.2f%%" % (self.min_iou, mAP_score*100.00))


    def __init__(self, config_path):
        self.evaluatex(config_path)

dir_path = os.path.dirname(os.path.realpath(__file__))
x = CalculateMAP(os.path.join(dir_path, "config.ini"))
