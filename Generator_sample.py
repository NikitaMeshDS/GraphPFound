import random

def sample_item(p_look=0.2, p_rel=0.3):
    """
    Один товар: (ci, look, rel)
    ci ∈ {0..5}
    look ∈ {0,1}
    rel ∈ {0,1}, но buy=1 возможно только если look=1
    """
    ci = random.randint(0, 5)
    look = 1 if random.random() < p_look else 0
    if look == 1 and random.random() < p_rel:
        rel = 1
    else:
        rel = 0
    return (ci, look, rel)


def generate_feed(n_pairs, p_look=0.2, p_rel=0.3):
    """
    Генератор выдачи из n_pairs состояний:
    каждый state = ((ci1, look1, rel1), (ci2, look2, rel2))
    """
    feed = []
    for _ in range(n_pairs):
        left  = sample_item(p_look, p_rel)
        right = sample_item(p_look, p_rel)
        feed.append((left, right))
    return feed


# пример
if __name__ == "__main__":
    res = generate_feed(10, p_look=0.3, p_rel=0.4)
    for pair in res:
        print(pair)