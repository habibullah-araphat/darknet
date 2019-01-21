def seg_intersec(a1, a2, b1, b2):
    if a1>b2 or a2<b1:
        return 0
    return min(b2, a2)-max(a1, b1)

def get_area(a):
    x1 = a['x']
    y1 = a['y']
    x2 = a['x'] + a['width']
    y2 = a['y'] + a['height']
    return (x2-x1)*(y2-y1)

def intersec_area(a, b):
    x1 = a['x']
    y1 = a['y']
    x2 = a['x']+a['width']
    y2 = a['y']+a['height']

    x3 = b['x']
    y3 = b['y']
    x4 = b['x']+b['width']
    y4 = b['y']+b['height']

    return seg_intersec(x1, x2, x3, x4)*seg_intersec(y1, y2, y3, y4)

def union_area(a, b):
    area_a = get_area(a)
    area_b = get_area(b)
    common_area = intersec_area(a, b)
    return area_a+area_b-common_area

def get_iou(a, b):
    u_area = union_area(a, b)
    i_area = intersec_area(a, b)
    if u_area==0:
        return 0.0
    return i_area/u_area