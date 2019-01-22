import tempfile, shutil
from unittest import TestCase
import os
from evaluate.mAP_utils import image_annotations
from PIL import Image

class TestImageAnnotations(TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def gen_image(self, width, height, img_path, ext):
        test_image = Image.new("RGB", (width, height))
        test_image.save(img_path, ext)

    def gen_txt(self, txt_path, input_list):
        with open(txt_path, 'w') as new_file:
            for input in input_list:
                new_file.write(input+"\n")

    def test_get_dim_can_return_dimentions(self):
        input_width = 223
        input_height = 555
        expected_width = 223
        expected_height = 555

        img_path = os.path.join(self.test_dir, "test.png")
        self.gen_image(input_width, input_height, img_path, "PNG")

        output_width, output_height = image_annotations.get_dim(img_path)
        self.assertEqual(output_width, expected_width, "width didn't match.")
        self.assertEqual(output_height, expected_height, "height didn't match.")

    def test_get_annotatoin_dict_should_return_expected_dict(self):
        input_text = "0    0.555 0.6666 0.20 0.40"
        input_width = 100
        input_height = 100
        expected_dict = {"class": 0, "x": 45, "y": 46, "width": 20, "height": 40}

        output_dict = image_annotations.get_annotatoin_dict(input_text, input_width, input_height)
        self.assertEqual(output_dict, expected_dict)

    def test_get_annotatoins_can_read_appropiate_annotation_list(self):
        input_width = 100
        input_height = 100
        input_annotation_list = ["0    0.555 0.6666 0.20 0.40", "1    0.05 0.07 0.04 0.06"]

        expected_annotations = [
            {"class": 0, "x": 45, "y": 46, "width": 20, "height": 40},
            {"class": 1, "x": 3, "y": 4, "width": 4, "height": 6}
        ]

        img_path = os.path.join(self.test_dir, "test.png")
        self.gen_image(input_width, input_height, img_path, "PNG")

        txt_path = os.path.join(self.test_dir, "test.txt")
        self.gen_txt(txt_path, input_annotation_list)

        output_annotations = image_annotations.get_annotatoins(img_path)
        self.assertEqual(output_annotations, expected_annotations)