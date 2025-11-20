from generate_chains import *

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

def extract_row_state(chain, row):
    """Возвращает (L_cb, R_cb) = (click,buy) для полки row"""
    L_cb = (0,0)
    R_cb = (0,0)
    for (r,c),_,click,purchase,_ in chain:
        if r == row:
            if c == 0:
                L_cb = (click,purchase)
            else:
                R_cb = (click,purchase)
    return L_cb, R_cb

def encode_row(L_cb, R_cb):
    return pair_to_x[(L_cb,R_cb)]

def infer_state(prev_col, exit_col, L_cb, R_cb):
    """
    prev_col: откуда пришли (0=лево,1=право)
    exit_col: куда ушли (0=лево,1=право)
    Если последняя строка, exit_col определяется по B/D на соответствующей колонке
    """
    if prev_col == 0 and exit_col == 0:
        return 0  # S0
    elif prev_col == 1 and exit_col == 0:
        return 1  # S1
    elif prev_col == 1 and exit_col == 1:
        return 2  # S2
    elif prev_col == 0 and exit_col == 1:
        return 3  # S3

def convert_chain_to_states(chain):
    results = []
    n = len(chain)

    # Сначала сгруппируем шаги по строкам
    rows = {}
    for idx,(state,color,click,purchase,brk) in enumerate(chain):
        row, col = state
        if row not in rows:
            rows[row] = []
        rows[row].append((idx,state,color,click,purchase,brk))

    prev_col = None
    sorted_rows = sorted(rows.keys())

    for i,row in enumerate(sorted_rows):
        steps = rows[row]

        # Определяем откуда пришли
        if prev_col is None:
            # стартовая колонка берем первую посещенную на этой строке
            first_step = steps[0]
            prev_col = first_step[1][1]

        # Определяем куда ушли с этой строки
        # Берем колонку последнего шага на строке
        last_step = steps[-1]
        exit_col = last_step[1][1]

        # Если последняя строка и есть B/D на какой-либо колонке, это финал
        final_suffix = ''
        L_cb, R_cb = extract_row_state(chain,row)
        if any(purchase==1 for idx,state,color,click,purchase,brk in steps):
            final_suffix = 'B'
        elif any(brk==1 for idx,state,color,click,purchase,brk in steps):
            final_suffix = 'D'

        x = encode_row(L_cb,R_cb)
        s = infer_state(prev_col, exit_col, L_cb, R_cb)

        if final_suffix:
            results.append(f"s{s}{final_suffix}: x{x}")
        else:
            results.append(f"s{s}: x{x}")

        prev_col = exit_col

    return results

n_chains = 1
chains = generate_chains(n_chains, 20, P_look, P_rel, P_break)

for chain in chains:
    convert_chain = convert_chain_to_states(chain)
    print(chain)
    print(convert_chain)
    
