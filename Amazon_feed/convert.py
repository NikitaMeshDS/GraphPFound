from generate_chains import *

def extract_row_state(chain, idx):
    """
    Возвращает (L_cb, R_cb) для строки chain[idx].row
    L_cb, R_cb = (click, buy)
    """

    (row_i, side_i) = chain[idx][0]  # state = (row, col)
    row = row_i

    # Инициализация
    L_cb = (0,0)
    R_cb = (0,0)

    # Найдём шаги на этой же строке
    same_row_steps = [ j for j,(st,_,c,p,b) in enumerate(chain)
                       if st[0] == row ]

    # смотрим назад
    for j in same_row_steps:
        (st, _, c, p, b) = chain[j]
        col = st[1]
        if j <= idx:
            if col == 0: L_cb = (c,p)
            else:        R_cb = (c,p)

    # смотрим вперёд — если позже будет посещена карточка, 
    # берём её (click,buy), потому что они достоверны
    for j in same_row_steps:
        if j > idx:
            (st,_,c,p,b) = chain[j]
            col = st[1]
            if col == 0: L_cb = (c,p)
            else:        R_cb = (c,p)

    return L_cb, R_cb


pair_to_x = {
    ((0,0),(0,0)): 0,
    ((1,0),(0,0)): 1,
    ((0,0),(1,0)): 2,
    ((1,0),(1,0)): 3,
    ((1,1),(0,0)): 4,
    ((0,0),(1,1)): 5,
    ((1,1),(1,0)): 6,
    ((1,0),(1,1)): 7,
}

def encode_row(L_cb, R_cb):
    return pair_to_x[(L_cb, R_cb)]

def infer_state(prev_state, current_state):
    """
    prev_state, current_state = (row, col)
    Возвращает s0..s3
    """

    if prev_state is None:
        # Начало траектории
        # Если пришел в (0,0) → сверху в левую → s0
        # Если пришел в (0,1) → сверху в правую → s3
        return 0 if current_state[1] == 0 else 3

    (pr, pc) = prev_state
    (cr, cc) = current_state

    # движение вниз
    if cr == pr + 1 and cc == pc:
        if cc == 0: return 0   # L column
        else:       return 2   # R column

    # движение вправо
    if cr == pr and cc == pc + 1:
        return 3

    # движение влево
    if cr == pr and cc == pc - 1:
        return 1


def convert_chain_to_states(chain):
    results = []
    prev_state = None

    for idx, (state, color, click, purchase, brk) in enumerate(chain):

        # вычисляем состояние строки по полной логике
        L_cb, R_cb = extract_row_state(chain, idx)
        x = encode_row(L_cb, R_cb)

        # движение s0..s3
        s = infer_state(prev_state, state)

        # финальные случаи
        if brk == 1:
            results.append(f"s{s}{'B' if purchase else 'D'}: x{x}")
            return results

        if purchase == 1:
            results.append(f"s{s}B: x{x}")
            return results

        # обычный шаг
        results.append(f"s{s}: x{x}")

        prev_state = state

    return results



n_chains = 1
chains = generate_chains(n_chains, 20, P_look, P_rel, P_break)
for chain in chains:
    convert_chain = convert_chain_to_states(chain)
    print(chain, convert_chain)