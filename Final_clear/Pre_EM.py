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

'''
S0-S3 и т д из hmm_matrix.py

Она зависит только от P_click!!!!!
N - ничего, C - клик, B - покупка
    R       L
X0  N       N
X1  C       N
X2  N       C
X3  C       C
X4  C,B     N
X5  N       C,B
X6  C,B     C
X7  C       C,B

для генерации будет удобен такой вариант отображения в виде кортежей
    R       L
X0  (0,0)     (0,0)
X1  (1,0)     (0,0)
X2  (0,0)     (1,0)
X3  (1,0)     (1,0)
X4  (1,1)     (0,0)
X5  (0,0)     (1,1)
X6  (1,1)     (1,0)
X7  (1,0)     (1,1)

matrix_click - семпл из нашей выдачи
вида
[
    [0.1, 0.2] i ая полка
]

если рассмотреть пример для S0 -> S3, когда мы начали только, то S0 первое ничего не должно выплюнуть
а вот после перехода когда мы сделали S3 мы можем узнать что было на полке 1

просто условно верояности клика 
'''

import numpy as np
def get_emission_matrix(matrix_click, p_break = 0.15):
    transition_matrix = np.zeros((13, 8))
    pcL = matrix_click[0, 0]
    pcR = matrix_click[0, 1]
    #S0 row
    transition_matrix[0, 0] = (1 - pcL)
    transition_matrix[0, 1] =  pcL 

    #S1 row
    transition_matrix[1, 0] = (1 - pcL) * (1 - pcR)
    transition_matrix[1, 1] = (1 - pcL) * pcR
    transition_matrix[1, 2] = pcL * (1 - pcR)
    transition_matrix[1, 3] = pcL * pcR

    #S2 row
    transition_matrix[2, 0] = (1 - pcR)
    transition_matrix[2, 2] = pcR

    # #S3 row
    transition_matrix[3, 0] = (1 - pcR) * (1 - pcL)
    transition_matrix[3, 1] = pcR * (1 - pcL)
    transition_matrix[3, 2] = (1 - pcR) * pcL
    transition_matrix[3, 3] = pcR * pcL

    #S0D
    transition_matrix[4, 0] = 1

    #S1D
    transition_matrix[5, 0] = (1 - pcL)
    transition_matrix[5, 2] = pcL

    #S2D
    transition_matrix[6, 0] = 1

    #S3D
    transition_matrix[7, 0] = (1 - pcR)
    transition_matrix[7, 1] = pcR

    #S0B row
    transition_matrix[8, 4] = 1

    #S1B row 
    transition_matrix[9, 4] = (1 - pcL)
    transition_matrix[9, 6] = pcL

    #S2B row
    transition_matrix[10, 5] = 1

    #S3B row
    transition_matrix[11, 5] = (1 - pcR)
    transition_matrix[11, 7] = pcR

    #NONE row
    transition_matrix[12, 0] = 1



    return transition_matrix

S_dict = {0: "S0", 1: "S1", 2: "S2", 3: "S3", 4: "S0B", 5: "S1B", 6: "S2B", 7: "S3B", 8: "S0D", 9: "S1D", 10: "S2D", 11: "S3D", 12: "None"}
X_dict = {0: "X0", 1: "X1", 2: "X2", 3: "X3", 4: "X4", 5: "X5", 6: "X6", 7: "X7"}
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

    return start_states.T

states_list = []

def demo(sostoyanie = 1): # sostoyanie - 1 полный вывод, 2 - прикольный вывод
    # Пример матрицы релевантностей (вероятность купить при просмотре)
    # 0-я строка — верхняя полка, 1-я — нижняя (условно); колонки — левая/правая карточки
    rel_matrix = np.array([
                            [0.2, 0.3],
                            [0.4, 0.1],
                            [0.6, 0.1],
                            [0.2, 0.7],
                            [0.01, 0.99],
                            [0.5, 0.5],
                            [0.3, 0.4],
                            [0.8, 0.2],
                            [0.15, 0.25],
                            [0.05, 0.95],
                            [0.45, 0.55],
                            [0.33, 0.66]
                            ]
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
    states_list.append(state[0]) 
    print("Начальное распределение состояний:\n", state)
    for i in range(rel_matrix.shape[0] - 1):
        T = get_transition_matrix(rel_matrix[i: i + 2, :], P_look, delivery[i:i+2, :], p_break=0.15)
        matrix_click = np.array(rel_matrix[i: i + 1, :])
        E = get_emission_matrix(matrix_click, p_break= 0.15)
        if i == 0:
            # print("\nМатрица переходов T (13x13) =\n", T)
            row_sums = T.sum(axis=1)
            print("\nСуммы по строкам:\n", row_sums)
        if sostoyanie == 1:
            print(f"matrix {i}:\n{rel_matrix[i: i + 2, :]}")

            state = state @ T

            print(f"\nРаспределение после {i} шага:\n", state)
            states_list.append(state[0])

            print("\nМатрица переходов T (13x13) =\n", T)

            print(f"\nМатрица эмиссий E (13x8) для шага {i}:\n", E)

            print()
        if sostoyanie == 2:
            print(f"matrix {i}:\n{rel_matrix[i: i + 2, :]}")

            state = state @ T
            print("\n=================================================")
            print(f"Распределение после {i} шага:")
            print("=================================================\n")
            for s in range(len(state)):
                if state[s] > 0.001:
                    print(f"P({S_dict[s]}) = {state[s]:.4f}")

            print(f"\nМатрица эмиссий E (13x8) для шага {i}:")
            for s in range(E.shape[0]):
                # выводим эмиссии только для «значащих» состояний S_k
                if state[s] <= 0.001:
                    continue

                row_vals = []
                for x in range(E.shape[1]):
                    if E[s, x] > 0.001:
                        row_vals.append(f"P({X_dict[x]}|{S_dict[s]})={E[s, x]:.4f}")
                if row_vals:
                    print(f"{S_dict[s]}: " + ", ".join(row_vals))
        
            print()
        

def GraphPfound(states):
    p_found = 0.0
    n_states = states.shape[0]
    for i in range(n_states):
        S_buy = states[i, 4:8]
        p_found += np.sum(S_buy)
    return p_found


if __name__ == "__main__":
    demo(sostoyanie=1)
    states_list = np.array(states_list)
    Pfound = GraphPfound(states_list)   
    print("GraphPfound =", Pfound)    