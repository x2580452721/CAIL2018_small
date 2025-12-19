# -*- coding: utf-8 -*-

TERM_BINS = ["death", "life", "0", "1-6", "7-12", "13-36", "37-60", "61-120", ">120"]

def term_to_label(term_obj: dict) -> int:
    """term_obj = meta['term_of_imprisonment']"""
    if term_obj.get("death_penalty", False):
        return 0
    if term_obj.get("life_imprisonment", False):
        return 1
    m = int(term_obj.get("imprisonment", 0))  # 通常按“月”
    if m == 0: return 2
    if 1 <= m <= 6: return 3
    if 7 <= m <= 12: return 4
    if 13 <= m <= 36: return 5
    if 37 <= m <= 60: return 6
    if 61 <= m <= 120: return 7
    return 8
