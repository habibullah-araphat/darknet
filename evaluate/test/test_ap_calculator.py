from unittest import TestCase
from evaluate import ap_calculator

class TestAPCalculator(TestCase):
    def setUp(self):
        # Create a temporary directory
        self.actual_dict = {
            "name.jpg": [
                {"class": 0, "x": 5, "y": 2, "width": 7, "height": 11},
                {"class": 0, "x": 7, "y": 11, "width": 7, "height": 11}
            ]
        }
        self.prediction_dict = {
            "name.jpg": [
                {"class": 0, "x": 5, "y": 2, "width": 7, "height": 11},
                {"class": 0, "x": 7, "y": 11, "width": 7, "height": 11}
            ]
        }
        self.class_name = 0
        self.min_iou = 0.5

    def test_total_occurance_sould_return_cnt_of_given_class(self):
        input_dict = self.actual_dict
        input_class = self.class_name
        expected_output = 2
        output = ap_calculator.total_occurance(input_dict, input_class)
        self.assertEqual(expected_output, output)

    def test_find_pred_sould_find_an_element(self):
        input_dict = self.actual_dict
        input_row = self.prediction_dict["name.jpg"][0]
        input_key = "name.jpg"
        expected_output = 1
        output = ap_calculator.find_pred(input_dict, input_row, input_key, self.min_iou)
        self.assertEqual(expected_output, output)

        remain_len = ap_calculator.total_occurance(input_dict, input_row["class"])
        expected_len = 1
        self.assertEqual(expected_len, remain_len)

    def test_calculate_ap_should_return_accurate_ap_score(self):
        expected_ap = 1.0
        ap = ap_calculator.calculate_ap(self.actual_dict, self.prediction_dict, self.class_name, self.min_iou)
        self.assertEqual(expected_ap, ap)


