from enum import IntEnum
import numpy as np
from collections import deque

class Color(IntEnum):
    GRAY  = 0
    GREEN = 1
    YELLOW = 2

def generate_delivery(n_rows, n_cols, color_probs, rng=None):
    """
    Генерирует выдачу (сетку полок) с цветами на основе Enum Color.

    color_probs: dict[Color, float], например:
        {Color.GRAY: 0.5, Color.GREEN: 0.3, Color.YELLOW: 0.2}
    """
    if rng is None:
        rng = np.random.default_rng()

    # список всех цветов в порядке Enum
    colors = np.array(list(Color))          # [Color.GRAY, Color.GREEN, Color.YELLOW]
    probs = np.array([color_probs[c] for c in colors], dtype=float)

    # на всякий случай нормируем (если суммы слегка "плывут")
    probs /= probs.sum()

    # выбираем индексы цветов (0..len(colors)-1) по заданному распределению
    idx = rng.choice(len(colors), size=(n_rows, n_cols), p=probs)

    # layout в виде Enum
    layout_enum = colors[idx]              # dtype=object или Color

    return layout_enum

def get_start_states(rel_matrix, p_look, delivery, p_break = 0.15):
    delivery = np.asarray(delivery)

    # приведение Enum -> int (если delivery из Enum), либо int -> int
    colors = np.vectorize(int)(delivery)
    # 0_0 0_1
    # 1_0 1_1
    #"S0", "S1", "S2", "S3", "S0B", "S1B", "S2B", "S3B", "S0D", "S1D", "S2D", "S3D",  "None"
    start_states = np.zeros((13, 1))

    p_move_right = p_look[colors[0, 0]][colors[0, 1]] / (p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]])
    p_move_left_down = p_look[colors[0, 0]][colors[1, 0]] / (p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]])
    start_states[0, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down # P(s0|s0)
    start_states[3, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break) # P(s3|s0)
    start_states[4, 0] = rel_matrix[0, 0] # P(s0b|s0)
    start_states[7, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1] # P(s3b|s0)
    start_states[8, 0] = (1 - rel_matrix[0, 0]) * p_break # P(s0d|s0)
    start_states[11, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break # P(s3d|s0)

    return start_states

def demo():
    # Пример матрицы релевантностей (вероятность купить при просмотре)
    # 0-я строка — верхняя полка, 1-я — нижняя (условно); колонки — левая/правая карточки
    rel_matrix = np.array([[0.2, 0.3],
                            [0.4, 0.1],
                            [0.6, 0.1],
                            [0.2, 0.7],
                            [0.01, 0.99],
                            [0.5, 0.5]]
    )

    P_look = np.array([
        #   GRAY   GREEN  YELLOW
        [ 0.6,  0.3,  0.1 ],  # GRAY
        [ 0.2,  0.6,  0.2 ],  # GREEN
        [ 0.3,  0.3,  0.4 ],  # YELLOW
    ])
    color_probs = {
        Color.GRAY: 0.5,
        Color.GREEN: 0.3,
        Color.YELLOW: 0.2,
    }

    delivery = generate_delivery(
        n_rows=rel_matrix.shape[0],
        n_cols=2,
        color_probs=color_probs
    )

    np.set_printoptions(precision=3, suppress=True)

    state = get_start_states(rel_matrix[0: 0 + 2, :], P_look, delivery[0:0+2, :], p_break=0.05)
    print("Start states:\n", state.T)

if __name__ == "__main__":
    demo()