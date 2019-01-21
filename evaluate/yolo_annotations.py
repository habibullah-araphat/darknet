import subprocess
import re

def parse_console_output(text):
    annotations = []
    str_list = text.split('\n')

    for cur_str in str_list:
        tokens = re.split('\W', cur_str)

        tmp_tokens = []
        for token in tokens:
            if token != '':
                tmp_tokens.append(token)

        tokens = tmp_tokens
        if len(tokens) == 10 and tokens[0] != "Loading":
            annotation = {
                            "class": tokens[0],
                            "x": int(tokens[3]),
                            "y": int(tokens[5]),
                            "width": int(tokens[7]),
                            "height": int(tokens[9]),
                          }
            annotations.append(annotation)
    return annotations

def get_annotation(binary_path, obj_path, cfg_path, weight_path, img_path):
    darket_command = 'cd {}; ./darknet detector test {} {} {} {} -ext_output'.format(binary_path, obj_path, cfg_path, weight_path, img_path)
    output = subprocess.check_output(darket_command, shell=True, stderr=subprocess.STDOUT)
    decoded_output = output.decode("utf-8")
    annotations = parse_console_output(decoded_output)
    return annotations
