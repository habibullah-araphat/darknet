import tempfile, shutil
import os
from unittest import TestCase
from evaluate import read_names

class TestLabelNames(TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_names_can_read_file(self):
        input_names = ["a class", "b class", "c class"]
        expected_names = ["a class", "b class", "c class"]

        with open(os.path.join(self.test_dir, 'names.txt'), 'w') as new_file:
            for name in input_names:
                new_file.write(name+"\n")

        file_path = os.path.join(self.test_dir, 'names.txt')
        result = read_names.get_names(file_path)
        self.assertEqual(result, expected_names)

    def test_get_names_can_trim_whitespace(self):
        input_names = ["a class ", "  b class", "  c class  "]
        expected_names = ["a class", "b class", "c class"]

        with open(os.path.join(self.test_dir, 'names.txt'), 'w') as new_file:
            for name in input_names:
                new_file.write(name+"\n")

        file_path = os.path.join(self.test_dir, 'names.txt')
        result = read_names.get_names(file_path)
        self.assertEqual(result, expected_names)

    def test_get_name_map_can_create_proper_name_dictionary(self):
        input_names = ["a class", "b class", "c class"]
        expected_output = {"a class": 0, "b class": 1, "c class": 2}

        result = read_names.get_name_map(input_names)
        self.assertEqual(result, expected_output)
