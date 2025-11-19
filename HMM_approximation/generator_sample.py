import random
import numpy as np
from hmm_Prel_Plook_ref import P_rel

def sample_item_color():
    """
    Генерирует случайный цвет товара.
    ci ∈ {0, 1, 2} - 3 цвета
    """
    return random.randint(0, 2)

def sample_item(p_rel_value):
    """
    Один товар: (ci, p_click, p_buy)
    ci ∈ {0, 1, 2} - цвет товара
    p_click = p_rel / 2
    p_buy = p_rel / 2
    """
    p_click = p_rel_value
    p_buy = p_rel_value
    return (p_click, p_buy)

def generate_feed(n_pairs):
    """
    Генератор выдачи из n_pairs пар товаров.
    Использует P_rel для вероятностей покупки каждого элемента.
    Возвращает: feed = [((ci1, p_click1, p_buy1), (ci2, p_click2, p_buy2)), ...]
    """
    feed = []
    
    for i in range(n_pairs):
        ci_left = sample_item_color()
        ci_right = sample_item_color()
        
        p_rel_left = P_rel[i % len(P_rel)][0]
        p_rel_right = P_rel[i % len(P_rel)][1]
        
        left_item = (ci_left,) + sample_item(p_rel_left)
        right_item = (ci_right,) + sample_item(p_rel_right)
        
        feed.append((left_item, right_item))
    
    return feed

if __name__ == "__main__":
    res = generate_feed(10)
    for pair in res:
        print(pair)