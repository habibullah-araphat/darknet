def get_names(file_path):
    res = []
    with open(file_path, 'r') as inp_file:
        for cur in inp_file:
            cur = cur.strip()
            if len(cur)>0:
                res.append(cur)
    return res

def get_name_map(name_list):
    res = dict()
    idx = 0
    for name in name_list:
        res[name] = idx
        idx = idx+1
    return res