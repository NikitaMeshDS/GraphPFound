import random
import numpy as np
from generator_sample import generate_feed
from hmm_Prel_Plook_ref import P_look, P_rel, P_break

def generate_chains(n_chains, n_pairs, P_look, P_rel, p_break=0.15):
    chains = []
    for _ in range(n_chains):
        feed = generate_feed(n_pairs)
        chain = generate_chain(feed, P_look, P_rel, p_break)
        chains.append(chain)
    return chains

#генерирует цепочку вида [(pair_idx, side), color, click, purchase, break]
def generate_chain(feed, P_look, P_rel, p_break):
    chain = []

    if random.random() < 0.5:
        initial_state = (0, 0)
    else:
        initial_state = (0, 1)
    if random.random() < p_break:
        chain.append((initial_state, feed[0][0][0], 0, 0, 1))
        return chain
    else:
        click = 0
        if random.random() < P_rel[0][0]:
            click = 1
            if random.random() < P_rel[0][0]:
                purchase = 1
                chain.append((initial_state, feed[0][0][0], click, purchase, 0))
                return chain
        chain.append((initial_state, feed[0][0][0], click, 0, 0))
        transit = 0
        state = initial_state
        while True:
            if state[0] == len(feed)-1:
                return chain
            if transit == 1:
                state = (state[0] + 1, state[1])
                transit = 0
            else:
                c_tec = feed[state[0]][state[1]][0]
                c_1, c_2 = feed[state[0]+1][state[1]][0], feed[state[0]][(state[1]+1)%2][0]
                    
                p_1 = P_look[c_tec][c_1]
                p_2 = P_look[c_tec][c_2]
                    
                total = p_1 + p_2
                if total > 0:
                    p_1_norm = p_1 / total
                else:
                    p_1_norm = 0.5
                    
                if random.random() < p_1_norm:
                    state = (state[0]+1, state[1])
                    transit = 0
                else:
                    state = (state[0], (state[1]+1)%2)
                    transit = 1
                
            p_rel = P_rel[state[0]][state[1]]
            color = feed[state[0]][state[1]][0]
            if random.random() < p_break:
                chain.append((state, color, 0, 0, 1))
                return chain
            else:
                if random.random() < p_rel:
                    click = 1
                    if random.random() < p_rel:
                        purchase = 1
                        chain.append((state, color, click, purchase, 0))
                        return chain
                    else:
                        chain.append((state, color, click, 0, 0))
                else:
                    chain.append((state, color, 0, 0, 0))
    

if __name__ == "__main__":
    n_chains = 10
    chains = generate_chains(n_chains, 20, P_look, P_rel, P_break)
    print(chains[0])