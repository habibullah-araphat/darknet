from unittest import TestCase
from evaluate.mAP_utils import iou_calculator


class TestIoUCalculator(TestCase):
    def test_seg_intersec_can_detectect_seg_correctly(self):
        input_a1 = 0
        input_a2 = 50
        input_b1 = 25
        input_b2 = 75
        expected_output = 25
        output = iou_calculator.seg_intersec(input_a1, input_a2, input_b1, input_b2)
        self.assertEqual(output, expected_output)

    def test_seg_intersec_for_invalid_input(self):
        input_a1 = 50
        input_a2 = 0
        input_b1 = 55
        input_b2 = 25
        expected_output = 0
        output = iou_calculator.seg_intersec(input_a1, input_a2, input_b1, input_b2)
        self.assertEqual(output, expected_output)

    def test_get_area_can_calculate_area_correctly(self):
        input = {"class": 0, "x": 5, "y":2, "width":7, "height":11}
        expected_output = 77
        output = iou_calculator.get_area(input)
        self.assertEqual(output, expected_output)

    def test_intersec_area_can_correctly_detect_the_intersection(self):
        input_a = {"class": 0, "x": 5, "y": 2, "width": 7, "height": 11}
        input_b = {"class": 0, "x": 7, "y": 11, "width": 70, "height": 110}
        expected_output = 10
        output = iou_calculator.intersec_area(input_a, input_b)
        self.assertEqual(expected_output, output)

    def test_union_area_can_correctly_detect_union(self):
        input_a = {"class": 0, "x": 5, "y": 2, "width": 7, "height": 11}
        input_b = {"class": 0, "x": 7, "y": 11, "width": 7, "height": 11}
        expected_output = 144
        output = iou_calculator.union_area(input_a, input_b)
        self.assertEqual(expected_output, output)

    def test_get_iou_can_calculate_iou_properly(self):
        input_a = {"class": 0, "x": 5, "y": 2, "width": 7, "height": 11}
        input_b = {"class": 0, "x": 7, "y": 11, "width": 7, "height": 11}
        expected_output = 0.06944444444444445
        output = iou_calculator.get_iou(input_a, input_b)
        self.assertEqual(expected_output, output)