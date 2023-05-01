def IoU(bbox_a, bbox_b) -> float:
    """
    Input:
        bbox : [top_left_x,top_left_y,width, height]
    """
    # check input
    assert len(bbox_a) == 4 and len(bbox_b) == 4, "bbox_a and bbox_b must have 4 elements"
    res: float = 0
    bbox_1 = {
        "left_x": bbox_a[0],
        "left_y": bbox_a[1],
        "right_x": bbox_a[0] + bbox_a[2],
        "right_y": bbox_a[1] + bbox_a[3],
    }
    bbox_1_area = bbox_a[2] * bbox_a[3]

    bbox_2 = {
        "left_x": bbox_b[0],
        "left_y": bbox_b[1],
        "right_x": bbox_b[0] + bbox_b[2],
        "right_y": bbox_b[1] + bbox_b[3],
    }
    bbox_2_area = bbox_b[2] * bbox_b[3]

    mid_width = min(bbox_1["right_x"], bbox_2["right_x"]) - max(bbox_1["left_x"], bbox_2["left_x"])
    mid_height = min(bbox_1["right_y"], bbox_2["right_y"]) - max(bbox_1["left_y"], bbox_2["left_y"])

    # if has zero or negative value
    if mid_width <= 0 or mid_height <= 0:
        return 0

    mid_area = mid_width * mid_height
    res = mid_area / (bbox_1_area + bbox_2_area - mid_area)
    return res
