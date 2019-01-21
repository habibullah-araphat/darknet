from evaluate import load_config
import tempfile, shutil
from unittest import TestCase
import os
import configparser

class TestLoadConfig(TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_get_config_can_load_config_properly(self):
        file_text = ''' [DEFAULT]
                        a = 45
                        b = yes/no.txt
                        c = 9 '''
        config_path = os.path.join(self.test_dir, "config.ini")

        with open(config_path, 'w') as file_ptr:
            file_ptr.write(file_text)

        expected_result = configparser.ConfigParser()
        expected_result["DEFAULT"] = {
                                        "a": 45,
                                        "b": "yes/no.txt",
                                        "c": 9
                                    }

        result = load_config.get_config(config_path)
        self.assertEqual(result, expected_result)