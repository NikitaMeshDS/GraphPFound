import random
from Generator_sample import generate_feed

def generate_user_chain(feed, p_break=0.15, start_side='left'):
    """
    Генерирует цепочку поведения пользователя по карточкам.
    
    Args:
        feed: список пар товаров из generate_feed
        p_break: вероятность прервать цепочку на каждом шаге
        start_side: начальная сторона ('left' или 'right')
    
    Returns:
        chain: список кортежей (pair_idx, side, item, action)
        где action: 'visit', 'purchase', 'break'
    """
    if not feed:
        return []
    
    visited_cards = set()
    visited_sides = set()
    chain = []
    current_pair_idx = 0
    current_side = start_side
    
    while True:
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
        
        visited_sides.add((current_pair_idx, current_side))
        item_result = (ci, look, rel)
        chain.append((current_pair_idx, current_side, item_result, 'visit'))
        
        if rel == 1:
            chain[-1] = (current_pair_idx, current_side, item_result, 'purchase')
            break
        
        if random.random() < p_break:
            chain[-1] = (current_pair_idx, current_side, item_result, 'break')
            break
        
        if (current_pair_idx, 'left') in visited_sides and (current_pair_idx, 'right') in visited_sides:
            visited_cards.add(current_pair_idx)
        
        next_move = choose_next_move(current_pair_idx, current_side, len(feed), visited_cards, visited_sides)
        
        if next_move is None:
            break
        
        if next_move[0] != current_pair_idx:
            visited_cards.add(current_pair_idx)
        
        current_pair_idx, current_side = next_move
    
    return chain

def choose_next_move(current_pair_idx, current_side, total_pairs, visited_cards, visited_sides):
    """
    Выбирает следующий ход: вниз или влево-вправо.
    С левой стороны: вниз (следующая карточка слева) или вправо (та же карточка справа)
    С правой стороны: вниз (следующая карточка справа) или влево (та же карточка слева)
    """
    possible_moves = []
    
    if current_side == 'left':
        if current_pair_idx + 1 < total_pairs and (current_pair_idx + 1) not in visited_cards:
            possible_moves.append((current_pair_idx + 1, 'left'))
        if (current_pair_idx, 'right') not in visited_sides:
            possible_moves.append((current_pair_idx, 'right'))
    else:
        if current_pair_idx + 1 < total_pairs and (current_pair_idx + 1) not in visited_cards:
            possible_moves.append((current_pair_idx + 1, 'right'))
        if (current_pair_idx, 'left') not in visited_sides:
            possible_moves.append((current_pair_idx, 'left'))
    
    if not possible_moves:
        return None
    
    return random.choice(possible_moves)

def generate_chains(n_chains, n_pairs, p_look=0.2, p_break=0.15):
    """
    Генерирует n_chains цепочек поведения пользователя.
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

