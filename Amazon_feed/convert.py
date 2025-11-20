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

def infer_state(prev_move, cur_col):
    """
    prev_move: 'down' or 'side'
    cur_col: 0 (L) or 1 (R)
    """
    if prev_move == 'side':
        # диагональный спуск
        return 1 if cur_col == 0 else 3
    else:
        # вертикальный спуск (включая старт)
        return 0 if cur_col == 0 else 2



def convert_chain_to_states(chain):
    results = []
    prev_pos = None
    prev_move = None   # 'down' или 'side'

    for idx, (state, color, click, purchase, brk) in enumerate(chain):

        cur_pos = state  # (row, col)
        row, col = cur_pos

        # определение движения
        if prev_pos is None:
            cur_move = 'down'     # старт — приход сверху
        else:
            pr, pc = prev_pos
            if row == pr + 1 and col == pc:

                cur_move = 'down'
            elif row == pr and abs(col - pc) == 1:
                cur_move = 'side'
            
            
        L_cb, R_cb = extract_row_state(chain, idx)
        x = encode_row(L_cb, R_cb)


        if cur_move == 'down':

            s = infer_state(prev_move, col)

            # финальные случаи
            if brk == 1:
                suffix = 'B' if purchase else 'D'
                results.append(f"s{s}{suffix}: x{x}")
                return results

            if purchase == 1:
                results.append(f"s{s}B: x{x}")
                return results

            # обычный вниз-ход
            results.append(f"s{s}: x{x}")
        # финальные случаи
        if brk == 1:
            s = infer_state(prev_move, col)
            suffix = 'B' if purchase else 'D'
            results.append(f"s{s}{suffix}: x{x}")
            return results

        if purchase == 1:
            s = infer_state(prev_move, col)
            results.append(f"s{s}B: x{x}")
            return results

        prev_move = cur_move
        prev_pos = cur_pos

    return results



n_chains = 1
chains = generate_chains(n_chains, 20, P_look, P_rel, P_break)


for chain in chains:
    convert_chain = convert_chain_to_states(chain)
    print(chain)
    print(convert_chain)
