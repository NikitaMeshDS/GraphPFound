'''
Использовал бета-распределение Beta(α, β) для вероятностей клика и покупки.
 Это априорное распределение на [0,1]: чем меньше α и больше β, 
 тем чаще вероятность тянется к 0; наоборот, большие α и маленькие β — к 1.

Как крутить параметры:

click_beta=(α_click, β_click): задаёт форму для вероятности клика. 
Например, (1, 7) даёт среднее 1/8 ≈ 0.125 и сильный перекос к редким кликам. 
Если хочешь чаще, увеличивай α или уменьшай β.
buy_beta=(α_buy, β_buy): то же для конверсии, но применяется только когда клик уже случился. 
Например, (1, 25) даёт ещё более низкие покупки.
Общее правило: среднее бета-распределения = α / (α + β).
Так что можешь прицелиться в нужную среднюю вероятность,
выбирая отношение α к β, а масштаб (α+β) задаёт «уверенность» — большие суммы → меньше разброса от среднего.
'''


import random
from typing import Iterable, Sequence, Tuple

State = Tuple[int, int, int]

def sample_item(
    rng: random.Random,
    click_beta: Tuple[float, float] = (1.0, 6.0),
    buy_beta: Tuple[float, float] = (1.0, 7.0),
) -> State:
    """One catalog item sampled from Beta-distributed click/buy probabilities."""

    ci = rng.randint(0, 5)
    click_prob = rng.betavariate(*click_beta)
    click = int(rng.random() < click_prob)

    buy = 0
    if click:
        buy_prob = rng.betavariate(*buy_beta)
        buy = int(rng.random() < buy_prob)
    return (ci, click, buy)


def generate_feed(
    n_pairs: int,
    click_beta: Tuple[float, float] = (1.0, 4.8),
    buy_beta: Tuple[float, float] = (1.0, 3.5),
    seed: int | None = None,
) -> list[Tuple[State, State]]:
    """Simulate a feed of (left, right) items with Beta-distributed clicks/purchases."""

    rng = random.Random(seed)
    feed = []
    for _ in range(n_pairs):
        left = sample_item(rng, click_beta=click_beta, buy_beta=buy_beta)
        right = sample_item(rng, click_beta=click_beta, buy_beta=buy_beta)
        feed.append((left, right))
    return feed


if __name__ == "__main__":
    res = generate_feed(100, seed=random.randint(0, 1000))
    for pair in res:
        print(pair)
