import tempfile, shutil
from unittest import TestCase
import os
from evaluate import yolo_annotations

class TestYoloAnnotations(TestCase):
    def test_get_annotation(self):
        binary_path = "./darknet_files/"
        obj_path = "obj.data"
        cfg_path = "pringles-big.cfg"
        weight_path = "pringles-big_11100.weights"
        img_path = "3.jpg"
        output = yolo_annotations.get_annotation(binary_path, obj_path, cfg_path, weight_path, img_path)
        pass

    def test_parse_console_output(self):
        binary_path = "./darknet_files/"
        obj_path = "obj.data"
        cfg_path = "pringles-big.cfg"
        weight_path = "pringles-big_11100.weights"
        img_path = "3.jpg"
        expected_result = [
            {'class': 'PO', 'height': 451, 'width': 144, 'x': 179, 'y': 25},
            {'class': 'PS', 'height': 446, 'width': 126, 'x': 328, 'y': 28}
        ]

        output = yolo_annotations.get_annotation(binary_path, obj_path, cfg_path, weight_path, img_path)
        result = yolo_annotations.parse_console_output(output)

        self.assertEqual(result, expected_result)
