import random

P_REL_VALUES = [0, 0.07, 0.14, 0.41, 0.61]

def predict_p_rel(ci):
    """
    Модель предсказывает p_rel на основе цвета ci.
    Возвращает одно из 5 значений: 0, 0.07, 0.14, 0.41, 0.61
    """
    mapping = {
        0: 0,
        1: 0.07,
        2: 0.14,
        3: 0.41,
        4: 0.61,
        5: 0.14
    }
    return mapping.get(ci, 0)

def sample_item(p_look=0.2):
    """
    Один товар: (ci, p_look, p_rel)
    ci ∈ {0..5}
    p_look: вероятность посмотреть на товар
    p_rel: вероятность покупки (предсказывается моделью на основе ci)
    """
    ci = random.randint(0, 5)
    p_rel = predict_p_rel(ci)
    return (ci, p_look, p_rel)

def generate_feed(n_pairs, p_look=0.2):
    """
    Генератор выдачи из n_pairs состояний:
    каждый state = ((ci1, p_look, p_rel1), (ci2, p_look, p_rel2))
    """
    feed = []
    for _ in range(n_pairs):
        left  = sample_item(p_look)
        right = sample_item(p_look)
        feed.append((left, right))
    return feed

if __name__ == "__main__":
    res = generate_feed(10, p_look=0.3)
    for pair in res:
        print(pair)