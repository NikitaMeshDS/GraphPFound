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


def create_p_look_matrix(P_look, delivery, start_pos=(0, 0)):
    """
    P_look: np.ndarray shape (n_colors, n_colors)
        P_look[c_from, c_to] — P(цвет_to | цвет_from)

    delivery: 2D-массив (n_rows x n_cols)
        коды цветов ячеек выдачи:
        - либо np.ndarray из IntEnum (Color),
        - либо np.ndarray из int (0..n_colors-1)

    start_pos: (r, c)
        координаты стартовой ячейки, для которой нет "предыдущей".
        Для неё можно считать P=1 (старт) или 0 — тут я ставлю 1.0.

    Возвращает:
        probs: 2D-массив той же формы, что delivery,
        где probs[r, c] = P(color(r,c) | color(parent(r,c)))
        по заданной матрице P_look.
    """
    delivery = np.asarray(delivery)
    n_rows, n_cols = delivery.shape

    # приведение Enum -> int (если delivery из Enum), либо int -> int
    colors = np.vectorize(int)(delivery)

    # задаём топологию — кто с кем сосед
    def neighbors(r, c):
        # крест: вверх, вниз, влево, вправо
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                yield nr, nc

    # 1. Строим "родителя" для каждой ячейки BFS-ом от start_pos
    parents = {start_pos: None}
    q = deque([start_pos])

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors(r, c):
            if (nr, nc) not in parents:
                parents[(nr, nc)] = (r, c)
                q.append((nr, nc))

    # 2. Считаем P(ячейка | родитель) по цветам
    probs = np.zeros_like(colors, dtype=float)

    for (r, c), parent in parents.items():
        if parent is None:
            # стартовая ячейка — как хочешь, я ставлю 1.0
            probs[r, c] = 1.0
        else:
            pr, pc = parent
            c_from = colors[pr, pc]   # цвет родителя
            c_to   = colors[r, c]     # цвет текущей ячейки
            probs[r, c] = P_look[c_from, c_to]

    return probs


def get_transition_matrix(rel_matrix, p_look_matrix, p_break = 0.15):
    # 0_0 0_1
    # 1_0 1_1
    #"S0", "S1", "S2", "S3", "S0B", "S1B", "S2B", "S3B", "S0D", "S1D", "S2D", "S3D",  "None"
    transition_matrix = np.zeros((13, 13))
    #first row
    p_move_right = p_look_matrix[0, 1] / (p_look_matrix[1, 0] + p_look_matrix[0, 1])
    p_move_left_down = p_look_matrix[1, 0] / (p_look_matrix[1, 0] + p_look_matrix[0, 1])
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
    p_move_left = p_look_matrix[0, 0] / (p_look_matrix[0, 0] + p_look_matrix[1, 1])
    p_move_right_down = p_look_matrix[1, 1] / (p_look_matrix[0, 0] + p_look_matrix[1, 1])
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

    layout_enum = generate_delivery(
        n_rows=6,
        n_cols=2,
        color_probs=color_probs
    )
    p_look_matrix = create_p_look_matrix(P_look, layout_enum)
    T = get_transition_matrix(rel_matrix, p_look_matrix, p_break=0.15)

    np.set_printoptions(precision=3, suppress=True)
    # print("\nМатрица переходов T (13x13) =\n", T)

    row_sums = T.sum(axis=1)
    print("\nСуммы по строкам:\n", row_sums)

    state = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    state = state @ T
    for i in range(1, rel_matrix.shape[0] - 1):
        m = np.vstack((rel_matrix[i], rel_matrix[i+1]))
        print(f"matrix {i}:\n{m}")
        T = get_transition_matrix(m, p_look_matrix[i:i+2, :], p_break=0.05)
        state = state @ T
        print(f"\nРаспределение после {i} шага:\n", state)
        print()

if __name__ == "__main__":
    demo()