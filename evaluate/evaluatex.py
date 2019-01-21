from evaluate import load_config
from evaluate import read_names
from evaluate import yolo_annotations
from evaluate import ap_calculator
from evaluate import image_annotations

profile = "DEFAULT"
class Evaluatex:
    def init_pars(self, config):
        self.darknet_dir = config[profile]["darknet_dir"]
        self.obj_path = config[profile]["obj_path"]
        self.names_path = config[profile]["names_path"]
        self.cfg_path = config[profile]["cfg_path"]
        self.weight_path = config[profile]["weight_path"]
        self.image_list_path = config[profile]["image_list_path"]

    def get_actual_dict(self, image_list):
        actual_dict = dict()
        for image in image_list:
            actual_dict[image] = image_annotations.get_annotatoins(image)
        return actual_dict

    def get_prediction_dict(self, image_list, names_dict):
        prediction_dict = dict()
        for image in image_list:
            print(image)
            prediction_dict[image] = yolo_annotations.get_annotation(self.darknet_dir, self.obj_path, self.cfg_path, self.weight_path, image)
            for idx in range(len(prediction_dict[image])):
                print("type idx:", type(idx))
                print(prediction_dict[image][idx])
                prediction_dict[image][idx]["class"] = names_dict[prediction_dict[image][idx]["class"]]
        return prediction_dict

    def evaluatex(self, config_path):
        config = load_config.get_config(config_path)
        self.init_pars(config)
        names_list = read_names.get_names(self.names_path)
        names_dict = read_names.get_name_map(names_list)
        image_list = read_names.get_names(self.image_list_path)

        actual_dict = self.get_actual_dict(image_list)
        prediction_dict = self.get_prediction_dict(image_list, names_dict)
        for idx in range(len(names_list)):
            cur_ap = ap_calculator.calculate_ap(actual_dict, prediction_dict, idx, 0.5)
            print(names_list[idx], cur_ap)

    def __init__(self, config_path):
        self.evaluatex(config_path)

x = Evaluatex("/home/konok/Desktop/darknet/evaluate/config.ini")
