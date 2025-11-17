import random

def sample_item(p_click=0.2, p_buy_given_click=0.3):
    """
    Один товар: (ci, click, buy)
    ci ∈ {0..5}
    click ∈ {0,1}
    buy ∈ {0,1}, но buy=1 возможно только если click=1
    """
    ci = random.randint(0, 5)
    click = 1 if random.random() < p_click else 0
    if click == 1 and random.random() < p_buy_given_click:
        buy = 1
    else:
        buy = 0
    return (ci, click, buy)


def generate_feed(n_pairs, p_click=0.2, p_buy_given_click=0.3):
    """
    Генератор выдачи из n_pairs состояний:
    каждый state = ((ci1, click1, buy1), (ci2, click2, buy2))
    """
    feed = []
    for _ in range(n_pairs):
        left  = sample_item(p_click, p_buy_given_click)
        right = sample_item(p_click, p_buy_given_click)
        feed.append((left, right))
    return feed


# пример
if __name__ == "__main__":
    res = generate_feed(10, p_click=0.3, p_buy_given_click=0.4)
    for pair in res:
        print(pair)