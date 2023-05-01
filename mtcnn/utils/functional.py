import random
from typing import List, Tuple


def random_picker(src: List, num: int) -> Tuple[List, List]:
    idx_list: set[int] = set()
    idx_max = len(src) - 1
    # do idx select
    while num > 0:
        select_idx = random.randint(0, idx_max)
        if select_idx not in idx_list:
            idx_list.add(select_idx)
            num -= 1
    # do list select
    select_list = [src[idx] for idx in idx_list]

    # do list remove
    removed_list = [src[idx] for idx in range(len(src)) if idx not in idx_list]

    return select_list, removed_list


def split_num(total_num, ratio: Tuple) -> Tuple:
    assert sum(ratio) == 1, "ratio must sum to 1"
    split_nums: list[int] = [int(total_num * ratio[i]) for i in range(len(ratio))]
    # check sum
    rest = total_num - sum(split_nums)
    if rest > 0:
        for i in range(rest):
            split_nums[i] += 1

    return tuple(split_nums)


def default_scale_step(stop: float, num: int, start: float = 1):
    """
    generate scale [start, stop]
    """
    # use a linear way to generate scale
    assert start > 0 and stop > 0 and num > 0, "start, stop, num must be positive"
    step = (stop - start) / (num - 1)

    while num > 0:
        yield start + (num - 1) * step
