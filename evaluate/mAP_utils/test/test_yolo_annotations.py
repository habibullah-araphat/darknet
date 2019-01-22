from unittest import TestCase
from evaluate.mAP_utils import yolo_annotations


class TestYoloAnnotations(TestCase):
    def test_run_binary(self):
        binary_path = "./darknet_files/"
        obj_path = "obj.data"
        cfg_path = "pringles-big.cfg"
        weight_path = "/home/konok/darknet/pringles-big-training/pringles-big_11100.weights"
        img_path = "3.jpg"
        output = yolo_annotations.run_binary(binary_path, obj_path, cfg_path, weight_path, img_path)
        expected_output = ""
        pass

    def test_get_annotation(self):
        binary_path = "./darknet_files/"
        obj_path = "obj.data"
        cfg_path = "pringles-big.cfg"
        weight_path = "/home/konok/darknet/pringles-big-training/pringles-big_11100.weights"
        img_path = "3.jpg"
        output = yolo_annotations.get_annotation(binary_path, obj_path, cfg_path, weight_path, img_path)
        expected_output = [{'class': 'PO', 'height': 451, 'width': 144, 'x': 179, 'y': 25},
                           {'class': 'PS', 'height': 446, 'width': 126, 'x': 328, 'y': 28}]
        self.assertEqual(expected_output, output)

    def test_parse_console_output(self):
        input_data = '''
                69 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
               70 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
               71 res   68                  19 x  19 x1024   ->    19 x  19 x1024
               72 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
               73 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
               74 res   71                  19 x  19 x1024   ->    19 x  19 x1024
               75 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
               76 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
               77 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
               78 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
               79 conv    512  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x 512  0.379 BFLOPs
               80 conv   1024  3 x 3 / 1    19 x  19 x 512   ->    19 x  19 x1024  3.407 BFLOPs
               81 conv     24  1 x 1 / 1    19 x  19 x1024   ->    19 x  19 x  24  0.018 BFLOPs
               82 yolo
               83 route  79
               84 conv    256  1 x 1 / 1    19 x  19 x 512   ->    19 x  19 x 256  0.095 BFLOPs
               85 upsample            2x    19 x  19 x 256   ->    38 x  38 x 256
               86 route  85 61
               87 conv    256  1 x 1 / 1    38 x  38 x 768   ->    38 x  38 x 256  0.568 BFLOPs
               88 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
               89 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
               90 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
               91 conv    256  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x 256  0.379 BFLOPs
               92 conv    512  3 x 3 / 1    38 x  38 x 256   ->    38 x  38 x 512  3.407 BFLOPs
               93 conv     24  1 x 1 / 1    38 x  38 x 512   ->    38 x  38 x  24  0.035 BFLOPs
               94 yolo
               95 route  91
               96 conv    128  1 x 1 / 1    38 x  38 x 256   ->    38 x  38 x 128  0.095 BFLOPs
               97 upsample            2x    38 x  38 x 128   ->    76 x  76 x 128
               98 route  97 36
               99 conv    128  1 x 1 / 1    76 x  76 x 384   ->    76 x  76 x 128  0.568 BFLOPs
              100 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
              101 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
              102 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
              103 conv    128  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 128  0.379 BFLOPs
              104 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
              105 conv     24  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x  24  0.071 BFLOPs
              106 yolo
            Loading weights from pringles-big-training/pringles-big_11100.weights...Done!
            test_images/3.jpg: Predicted in 19.500830 seconds.
            PringlesOriginal: 100%   (left_x:  179   top_y:  25   width:  144   height:  451)
            PringlesSourCreamandOnion: 100%   (left_x:  328   top_y:  28   width:  126   height:  446)
        '''

        output = yolo_annotations.parse_console_output(input_data)
        expected_output = [{'class': 'PringlesOriginal', 'height': 451, 'width': 144, 'x': 179, 'y': 25},
                           {'class': 'PringlesSourCreamandOnion', 'height': 446, 'width': 126, 'x': 328, 'y': 28}]

        self.assertEqual(expected_output, output)
