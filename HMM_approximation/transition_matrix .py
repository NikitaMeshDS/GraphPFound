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


def get_transition_matrix(rel_matrix, p_look, delivery, p_break = 0.15):
    """
    rel_matrix: np.ndarray shape (n_rels, 2)
        rel_matrix — P(релевантности) для ячеек
    
    P_look: np.ndarray shape (n_colors, n_colors)
        P_look[c_from, c_to] — P(цвет_to | цвет_from)

    delivery: 2D-массив (n_rows x n_cols)
        коды цветов ячеек выдачи:
        - либо np.ndarray из IntEnum (Color),
        - либо np.ndarray из int (0..n_colors-1)
    """
    delivery = np.asarray(delivery)

    # приведение Enum -> int (если delivery из Enum), либо int -> int
    colors = np.vectorize(int)(delivery)
    # 0_0 0_1
    # 1_0 1_1
    #"S0", "S1", "S2", "S3", "S0B", "S1B", "S2B", "S3B", "S0D", "S1D", "S2D", "S3D",  "None"
    transition_matrix = np.zeros((13, 13))
    #first row
    p_move_right = p_look[colors[0, 0]][colors[0, 1]] / (p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]])
    p_move_left_down = p_look[colors[0, 0]][colors[1, 0]] / (p_look[colors[0, 0]][colors[0, 1]] + p_look[colors[0, 0]][colors[1, 0]])
    transition_matrix[0, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down # P(s0|s0)
    transition_matrix[0, 3] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break) # P(s3|s0)
    transition_matrix[0, 4] = rel_matrix[0, 0] # P(s0b|s0)
    transition_matrix[0, 7] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1] # P(s3b|s0)
    transition_matrix[0, 8] = (1 - rel_matrix[0, 0]) * p_break # P(s0d|s0)
    transition_matrix[0, 11] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break # P(s3d|s0)

    #second_row
    transition_matrix[1, 0] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_left_down # P(s0|s1)
    transition_matrix[1, 3] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * (1 - p_break) # P(s3|s1)
    transition_matrix[1, 4] = rel_matrix[0, 0] # P(s0b|s1)
    transition_matrix[1, 7] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * rel_matrix[0, 1] # P(s3b|s1)
    transition_matrix[1, 8] = (1 - rel_matrix[0, 0]) * p_break # P(s0d|s1)
    transition_matrix[1, 11] = (1 - rel_matrix[0, 0]) * (1 - p_break) * p_move_right * (1 - rel_matrix[0, 1]) * p_break # P(s3d|s1)

    #third row
    p_move_left = p_look[colors[0, 1]][colors[0, 0]] / (p_look[colors[0, 1]][colors[0, 0]] + p_look[colors[0, 1]][colors[1, 1]])
    p_move_right_down = p_look[colors[0, 1]][colors[1, 1]] / (p_look[colors[0, 1]][colors[0, 0]] + p_look[colors[0, 1]][colors[1, 1]])
    transition_matrix[2, 1] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break) #P(s1|s2)
    transition_matrix[2, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down # P(s2|s2)
    transition_matrix[2, 5] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0] #P(s1b|s2)
    transition_matrix[2, 6] = rel_matrix[0, 1] # P(s2b|s2)
    transition_matrix[2, 9] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break # P(s1d|s2)
    transition_matrix[2, 10] = (1 - rel_matrix[0, 1]) * p_break # P(s2d|s2)

    #fourth row
    transition_matrix[3, 1] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * (1 - p_break) #P(s1|s3)
    transition_matrix[3, 2] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_right_down # P(s2|s3)
    transition_matrix[3, 5] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * rel_matrix[0, 0] #P(s1b|s3)
    transition_matrix[3, 6] = rel_matrix[0, 1] # P(s2b|s3)
    transition_matrix[3, 9] = (1 - rel_matrix[0, 1]) * (1 - p_break) * p_move_left * (1 - rel_matrix[0, 0]) * p_break # P(s1d|s3)
    transition_matrix[3, 10] = (1 - rel_matrix[0, 1]) * p_break # P(s2d|s3)

    #rest_rows
    transition_matrix[4:, 12] = 1.0  # P(None|sX) = 1.0 for all buying and break states

    return transition_matrix

from enum import IntEnum


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
        n_rows=6,
        n_cols=2,
        color_probs=color_probs
    )

    np.set_printoptions(precision=3, suppress=True)

    state = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(rel_matrix.shape[0] - 1):
        T = get_transition_matrix(rel_matrix[i: i + 2, :], P_look, delivery[i:i+2, :], p_break=0.05)
        if i == 0:
            # print("\nМатрица переходов T (13x13) =\n", T)
            row_sums = T.sum(axis=1)
            print("\nСуммы по строкам:\n", row_sums)
        print(f"matrix {i}:\n{rel_matrix[i: i + 2, :]}")
        state = state @ T
        print(f"\nРаспределение после {i} шага:\n", state)
        print()
    

if __name__ == "__main__":
    demo()