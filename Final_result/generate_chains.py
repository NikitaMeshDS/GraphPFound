import random
import numpy as np
from generator_sample import generate_feed
from hmm_Prel_Plook_ref import P_look, P_rel, P_break

def generate_chains(n_chains, n_pairs, P_look, P_rel, p_break=0.15):
    chains = []
    for _ in range(n_chains):
        feed = generate_feed(n_pairs)
        chain = generate_chain(feed, P_look, P_rel, p_break)
        chains.append(chain)
    return chains

#генерирует цепочку вида [(pair_idx, side), color, click, purchase, break]
def generate_chain(feed, P_look, P_rel, p_break):
    """
    Генерирует цепочку вида:
      [( (row, col), color_id, click, purchase, break ), ... ]
    где color_id мы дополнительно маппим в индексы 0..2 для обращения к P_look.
    """
    chain = []

    # количество цветов, для которых определён P_look
    n_colors = len(P_look)  # 3

    # --- стартовое состояние: либо (0,0), либо (0,1) ---
    if random.random() < 0.5:
        initial_state = (0, 0)
    else:
        initial_state = (0, 1)

    # цвет стартовой карточки из feed
    raw_color0 = feed[0][initial_state[1]][0]
    c0 = raw_color0 % n_colors  # приводим к 0..2

    # --- сразу break с первой карточки ---
    if random.random() < p_break:
        chain.append((initial_state, c0, 0, 0, 1))
        return chain
    else:
        click = 0
        purchase = 0

        # вероятность клика по P_rel для первой строки
        p_rel0 = P_rel[0][initial_state[1]]
        if random.random() < p_rel0:
            click = 1
            # тут можно было бы использовать P_buy, но у тебя пока P_rel:
            if random.random() < p_rel0:
                purchase = 1
                chain.append((initial_state, c0, click, purchase, 0))
                return chain

        chain.append((initial_state, c0, click, purchase, 0))

        transit = 0
        state = initial_state

        # --- основной цикл переходов ---
        while True:
            row, col = state

            # если на последней строке — выходим (дальше полок нет)
            if row == len(feed) - 1:
                return chain

            if transit == 1:
                # спускаемся вниз по той же колонке
                state = (row + 1, col)
                transit = 0
            else:
                # смотрим на текущий цвет и кандидатов для перехода
                raw_c_tec = feed[row][col][0]
                raw_c_down = feed[row + 1][col][0]
                raw_c_side = feed[row][(col + 1) % 2][0]

                c_tec = raw_c_tec % n_colors
                c_1 = raw_c_down % n_colors
                c_2 = raw_c_side % n_colors

                p_1 = P_look[c_tec][c_1]
                p_2 = P_look[c_tec][c_2]

                total = p_1 + p_2
                if total > 0:
                    p_1_norm = p_1 / total
                else:
                    p_1_norm = 0.5  # если матрица даёт 0, делим пополам

                if random.random() < p_1_norm:
                    # идём вниз
                    state = (row + 1, col)
                    transit = 0
                else:
                    # идём в соседнюю колонку, остаёмся на той же строке
                    state = (row, (col + 1) % 2)
                    transit = 1

            row, col = state
            raw_color = feed[row][col][0]
            color_id = raw_color % n_colors

            # вероятность релевантности на этой позиции
            p_rel = P_rel[row][col]

            # break
            if random.random() < p_break:
                chain.append((state, color_id, 0, 0, 1))
                return chain
            else:
                click = 0
                purchase = 0
                if random.random() < p_rel:
                    click = 1
                    # снова «покупка при клике» через p_rel, как у тебя
                    if random.random() < p_rel:
                        purchase = 1
                        chain.append((state, color_id, click, purchase, 0))
                        return chain

                chain.append((state, color_id, click, purchase, 0))
    

if __name__ == "__main__":
    n_chains = 10
    chains = generate_chains(n_chains, 20, P_look, P_rel, P_break)
    print(chains)