# HMM_approximation/build_em_seq.py

import numpy as np
from generate_chains import generate_chains
from hmm_Prel_Plook_ref import P_look, P_rel, P_break  # твои матрицы

# кодировки X0..X7 для паттернов (click,buy) по (L,R)
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

# карта состояний S0..S3, S0B..S3B, S0D..S3D
# (мы будем её использовать ровно так, как договорились раньше)
state_name_to_idx = {
    "S0": 0, "S1": 1, "S2": 2, "S3": 3,
    "S0B": 4, "S1B": 5, "S2B": 6, "S3B": 7,
    "S0D": 8, "S1D": 9, "S2D": 10, "S3D": 11,
    "None": 12
}

def extract_row_state(chain, row):
    """Возвращает (L_cb, R_cb) = (click,buy) для полки row"""
    L_cb = (0, 0)
    R_cb = (0, 0)
    for (r, c), color, click, purchase, brk in chain:
        if r == row:
            if c == 0:
                L_cb = (click, purchase)
            else:
                R_cb = (click, purchase)
    return L_cb, R_cb

def encode_row(L_cb, R_cb):
    """Кодируем пару (L_cb,R_cb) в X0..X7 (0..7)."""
    return pair_to_x[(L_cb, R_cb)]

def infer_state(prev_col, exit_col, suffix):
    """
    Определяем S0/S1/S2/S3 и добавляем B/D-суффикс, если нужно.
    prev_col: откуда пришли (0=лево,1=право)
    exit_col: куда ушли (0=лево,1=право)
    suffix: '', 'B' или 'D'
    """
    if prev_col == 0 and exit_col == 0:
        base = "S0"
    elif prev_col == 1 and exit_col == 0:
        base = "S1"
    elif prev_col == 1 and exit_col == 1:
        base = "S2"
    elif prev_col == 0 and exit_col == 1:
        base = "S3"
    else:
        # теоретически не должно сюда попадать
        base = "None"

    name = base + suffix
    if name not in state_name_to_idx:
        name = base  # на всякий случай

    return state_name_to_idx[name]

def convert_chain_to_z_o(chain):
    """
    На вход: одна цепочка [( (row,col), color, click, purchase, break ), ...]
    На выход:
      z_seq: np.array формы (T,) — индексы состояний 0..12
      o_seq: np.array формы (T,) — индексы наблюдений 0..7
    """
    rows = {}
    for idx, (state, color, click, purchase, brk) in enumerate(chain):
        row, col = state
        rows.setdefault(row, []).append((idx, state, color, click, purchase, brk))

    sorted_rows = sorted(rows.keys())
    z_list = []
    o_list = []

    prev_col = None

    for i, row in enumerate(sorted_rows):
        steps = rows[row]

        # откуда пришли
        if prev_col is None:
            first_step = steps[0]
            prev_col = first_step[1][1]

        # куда ушли (последняя колонка на этой строке)
        last_step = steps[-1]
        exit_col = last_step[1][1]

        # суффикс B/D (если на этой строке была покупка или break)
        suffix = ''
        L_cb, R_cb = extract_row_state(chain, row)
        if any(purchase == 1 for _, st, _, _, purchase, brk in steps):
            suffix = 'B'
        elif any(brk == 1 for _, st, _, _, purchase, brk in steps):
            suffix = 'D'

        x = encode_row(L_cb, R_cb)            # наблюдение Xk → 0..7
        s_idx = infer_state(prev_col, exit_col, suffix)  # состояние Sk → 0..12

        z_list.append(s_idx)
        o_list.append(x)

        prev_col = exit_col

    return np.array(z_list, dtype=int), np.array(o_list, dtype=int)

def build_em_sequences(n_chains=5, n_pairs=20):
    """
    Генерирует n_chains цепочек и для каждой строит (z_seq, o_seq).

    Возвращает:
      list_z: список np.array, каждый длиной T_k
      list_o: список np.array, каждый длиной T_k
    """
    chains = generate_chains(n_chains, n_pairs, P_look, P_rel, P_break)
    list_z = []
    list_o = []

    for idx, chain in enumerate(chains, start=1):
        z_seq, o_seq = convert_chain_to_z_o(chain)
        list_z.append(z_seq)
        list_o.append(o_seq)

        # для контроля — можно оставить/убрать
        print(f"\n=== Цепочка {idx} ===")
        for step in chain:
            print(step)
        print("z_seq:", z_seq)
        print("o_seq:", o_seq)
        print("Длина цепочки:", len(o_seq))

    total_len = sum(len(o) for o in list_o)
    print("\n--- Сводка по сгенерированным последовательностям ---")
    print("Длины o_seq по цепочкам:", [len(o) for o in list_o])
    print("Суммарное число наблюдений:", total_len)

    return list_z, list_o

if __name__ == "__main__":
    # Тестовый запуск
    list_z, list_o = build_em_sequences(n_chains=5, n_pairs=20)
