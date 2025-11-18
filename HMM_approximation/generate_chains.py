import random
import numpy as np
from Generator_sample import generate_feed
from hmm_matrix import P, pi as initial_pi, states

def state_to_position(state, current_pair_idx, current_side):
    """
    Преобразует состояние в следующую позицию (pair_idx, side).
    S0: (i, left) -> (i+1, left)
    S1: (i, left) -> (i+1, right) или (i, left) -> (i, right) -> (i+1, right)
    S2: (i, right) -> (i+1, right)
    S3: (i, right) -> (i+1, left) или (i, right) -> (i, left) -> (i+1, left)
    S4: (i, left) -> (i+1, left) и остановка
    S5: (i, left) -> (i, right) и остановка
    S6: (i, right) -> (i+1, right) и остановка
    S7: (i, right) -> (i, left) и остановка
    """
    if state == "S0":
        return (current_pair_idx + 1, 'left'), False
    elif state == "S1":
        if random.random() < 0.5:
            return (current_pair_idx, 'right'), False
        else:
            return (current_pair_idx + 1, 'right'), False
    elif state == "S2":
        return (current_pair_idx + 1, 'right'), False
    elif state == "S3":
        if random.random() < 0.5:
            return (current_pair_idx, 'left'), False
        else:
            return (current_pair_idx + 1, 'left'), False
    elif state == "S4":
        return (current_pair_idx + 1, 'left'), True
    elif state == "S5":
        return (current_pair_idx, 'right'), True
    elif state == "S6":
        return (current_pair_idx + 1, 'right'), True
    elif state == "S7":
        return (current_pair_idx, 'left'), True
    return (current_pair_idx, current_side), True

def generate_user_chain(feed, p_break=0.15, start_side='left'):
    """
    Генерирует цепочку поведения пользователя по карточкам используя матрицу переходов P.
    
    Args:
        feed: список пар товаров из generate_feed
        p_break: вероятность прервать цепочку на каждом шаге (используется для остановки)
        start_side: начальная сторона ('left' или 'right')
    
    Returns:
        chain: список кортежей (pair_idx, side, item, action)
        где action: 'visit', 'purchase', 'break'
    """
    if not feed:
        return []
    
    state_to_idx = {state: i for i, state in enumerate(states)}
    idx_to_state = {i: state for i, state in enumerate(states)}
    
    current_state_idx = np.random.choice(4, p=initial_pi[:4] / initial_pi[:4].sum())
    current_state = idx_to_state[current_state_idx]
    
    current_pair_idx = 0
    current_side = 'left' if current_state in ["S0", "S1"] else 'right'
    
    chain = []
    visited_cards = set()
    
    while True:
        if current_state == "DEAD":
            break
        
        if current_pair_idx >= len(feed):
            break
        
        if current_pair_idx in visited_cards:
            break
        
        pair = feed[current_pair_idx]
        left_item, right_item = pair
        current_item = left_item if current_side == 'left' else right_item
        
        ci, p_look, p_rel = current_item
        
        look = 1 if random.random() < p_look else 0
        rel = 0
        if look == 1 and random.random() < p_rel:
            rel = 1
        
        item_result = (ci, look, rel)
        
        is_stopped = current_state in ["S4", "S5", "S6", "S7"]
        
        if rel == 1:
            action = 'purchase'
        elif is_stopped:
            action = 'break'
        else:
            action = 'visit'
        
        chain.append((current_pair_idx, current_side, item_result, action))
        visited_cards.add(current_pair_idx)
        
        if rel == 1 or is_stopped:
            break
        
        next_pos, will_stop = state_to_position(current_state, current_pair_idx, current_side)
        next_pair_idx, next_side = next_pos
        
        if next_pair_idx >= len(feed):
            break
        
        if will_stop:
            if current_state == "S0":
                current_state = "S4"
            elif current_state == "S1":
                current_state = "S5"
            elif current_state == "S2":
                current_state = "S6"
            elif current_state == "S3":
                current_state = "S7"
        else:
            next_state_probs = P[current_state_idx]
            current_state_idx = np.random.choice(len(states), p=next_state_probs)
            current_state = idx_to_state[current_state_idx]
        
        current_pair_idx, current_side = next_pair_idx, next_side
    
    return chain

def generate_chains(n_chains, n_pairs, p_look=0.2, p_break=0.15):
    """
    Генерирует n_chains цепочек поведения пользователя используя матрицу переходов P.
    """
    feed = generate_feed(n_pairs, p_look)
    chains = []
    
    for _ in range(n_chains):
        start_side = random.choice(['left', 'right'])
        chain = generate_user_chain(feed, p_break, start_side)
        chains.append(chain)
    
    return feed, chains

if __name__ == "__main__":
    feed, chains = generate_chains(n_chains=5, n_pairs=10, p_look=0.3, p_break=0.15)
    
    print("Feed:")
    for i, pair in enumerate(feed):
        print(f"  {i}: {pair}")
    
    print("\nChains:")
    for i, chain in enumerate(chains):
        print(f"\nChain {i+1}:")
        for pair_idx, side, item, action in chain:
            ci, look, rel = item
            print(f"  Pair {pair_idx}, {side}: ci={ci}, look={look}, rel={rel}, action={action}")

