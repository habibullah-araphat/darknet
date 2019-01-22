from evaluate.mAP_utils import iou_calculator


def total_occurance(actual_dict, class_name):
    total_cnt = 0
    for key, value in actual_dict.items():
        for element in value:
            if element["class"]==class_name:
                total_cnt += 1
    return total_cnt

def find_pred(actual_dict, target_row, dict_key, min_iou):
    if dict_key in actual_dict:
        for idx, cur_row in enumerate(actual_dict[dict_key]):
            if cur_row["class"]==target_row["class"]:
                iou_val = iou_calculator.get_iou(cur_row, target_row)
                if iou_val>=min_iou:
                    del actual_dict[dict_key][idx]
                    return 1
    return -1

def calculate_ap(actual_dict, prediction_dict, class_name, min_iou=0.5):
    recall_pres_pairs = []
    total = total_occurance(actual_dict, class_name)
    true_pos = 0
    false_pos = 0
    if total==0:
        return 0.0
    for key, value in prediction_dict.items():
        for cur_row in value:
            if cur_row['class']==class_name:
                pred_res = find_pred(actual_dict, cur_row, key, min_iou)
                if pred_res==1:
                    true_pos += 1
                else:
                    false_pos += 1
                recall_pres_pairs.append((true_pos/total, true_pos/(true_pos+false_pos)))

    recall_pres_pairs.sort()
    steps_val = [0.00]*11
    steps_idx = 10
    cur_mx = 0.00
    for cur_pair in recall_pres_pairs[::-1]:
        while steps_idx>=0 and 0.01*steps_idx>cur_pair[0]:
            steps_val[steps_idx] = cur_mx
            steps_idx -= 1
        cur_mx = max(cur_mx, cur_pair[1])
        steps_val[steps_idx] = cur_mx

    while steps_idx>=0:
        steps_val[steps_idx] = cur_mx
        steps_idx -= 1

    totoal_precision = sum(steps_val)
    return totoal_precision/11.00
