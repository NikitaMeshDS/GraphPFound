import random
import os
from feed_matrix_generator import generate_feed_matrix, load_matrix, save_matrix
from load_data import load_all

def generate_chains(n_chains, matrix, p_break=0.15):
    """Генерирует n_chains цепочек поведения пользователя"""
    chains = []
    for _ in range(n_chains):
        chain = generate_chain(matrix, p_break)
        if chain:
            chains.append(chain)
    return chains

def generate_chain(matrix, p_break=0.15):
    """Генерирует одну цепочку вида [(row, col), id, click, purchase, break]"""
    chain = []
    visited = set()
    rows = len(matrix)
    cols = len(matrix[0]) if matrix and matrix[0] else 0
    
    if rows == 0 or cols == 0:
        return chain
    
    state = (0, random.randint(0, cols - 1) if cols > 1 else 0)
    
    if matrix[state[0]][state[1]] is None:
        return chain
    
    visited.add(state)
    
    while True:
        item = matrix[state[0]][state[1]]
        
        if random.random() < p_break:
            chain.append((state, item['id'], 0, 0, 1))
            return chain
        
        click = 1 if random.random() < item['P_click'] else 0
        purchase = 1 if (click and random.random() < item['P_buy']) else 0
        
        chain.append((state, item['id'], click, purchase, 0))
        
        if purchase:
            return chain
        
        available_moves = _get_available_moves(state, matrix, rows, cols, visited)
        
        if not available_moves:
            return chain
        
        state = _choose_next_move(available_moves)
        visited.add(state)


def _get_available_moves(state, matrix, rows, cols, visited):
    """Возвращает список доступных переходов [(next_state, probability)]"""
    row, col = state
    moves = []
    
    if row < rows - 1:
        next_state = (row + 1, col)
        if matrix[row + 1][col] is not None and next_state not in visited:
            moves.append((next_state, matrix[row][col]['P_look1']))
    
    side_moves = []
    if col > 0:
        next_state = (row, col - 1)
        if matrix[row][col - 1] is not None and next_state not in visited:
            side_moves.append(next_state)
    
    if col < cols - 1:
        next_state = (row, col + 1)
        if matrix[row][col + 1] is not None and next_state not in visited:
            side_moves.append(next_state)
    
    if side_moves:
        p_look2 = matrix[row][col]['P_look2'] / len(side_moves)
        moves.extend([(s, p_look2) for s in side_moves])
    
    return moves


def _choose_next_move(available_moves):
    """Выбирает следующий шаг на основе вероятностей"""
    total_prob = sum(prob for _, prob in available_moves)
    
    if total_prob > 0:
        r = random.random() * total_prob
        cumulative = 0
        for next_state, prob in available_moves:
            cumulative += prob
            if r <= cumulative:
                return next_state
    
    return available_moves[0][0]


if __name__ == "__main__":
    matrix_filepath = "saved_matrix.json"
    
    if os.path.exists(matrix_filepath):
        print(f"Загрузка сохраненной матрицы из {matrix_filepath}...")
        matrix, metadata = load_matrix(matrix_filepath)
    else:
        print("Сохраненная матрица не найдена. Генерация новой матрицы...")
        print("Загрузка данных и моделей...")
        data, tifd, tf, w2v_model = load_all()
        
        query = "black T-shirt"
        rows, cols = 8, 2
        
        matrix = generate_feed_matrix(
            query=query,
            n_products=rows * cols,
            vectorizer=tifd,
            product_vectors=tf,
            data_df=data,
            rows=rows,
            cols=cols,
            w2v_model=w2v_model
        )
        
        save_matrix(matrix, matrix_filepath, query=query, rows=rows, cols=cols)
    
    chains = generate_chains(5, matrix, p_break=0.05)
    
    print(f"\nСгенерировано цепочек: {len(chains)}")
    if chains:
        print(f"\nПервая цепочка (длина {len(chains[0])}):")
        for step in chains[0]:
            print(f"  {step}")