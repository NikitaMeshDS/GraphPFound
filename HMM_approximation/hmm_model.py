import numpy as np
from collections import Counter
from generate_chains import generate_chains
from hmm_matrix import P, pi as initial_pi

class UserBehaviorHMM:
    def __init__(self, n_pairs, max_pairs=None):
        self.n_pairs = n_pairs
        self.max_pairs = max_pairs or n_pairs
        
        self.hidden_states = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "DEAD"]
        self.state_to_idx = {state: i for i, state in enumerate(self.hidden_states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.hidden_states)}
        
        self.observations = ['visit', 'click', 'purchase', 'break']
        self.obs_to_idx = {obs: i for i, obs in enumerate(self.observations)}
        self.idx_to_obs = {i: obs for obs, i in self.obs_to_idx.items()}
        
        self.n_states = len(self.hidden_states)
        self.n_obs = len(self.observations)
        
        self.pi = None
        self.A = None
        self.B = None
        
    def _determine_transition_state(self, prev_pos, curr_pos, next_pos, is_stopped):
        """
        Определяет состояние на основе перехода между позициями.
        
        S0: переход вниз по левой стороне (i, left) -> (i+1, left)
        S1: переход вправо и вниз с левой полосы (i, left) -> (i, right) -> (i+1, right) или (i, left) -> (i+1, right)
        S2: переход вниз по правой полосе (i, right) -> (i+1, right)
        S3: переход влево и вниз с правой полосы (i, right) -> (i, left) -> (i+1, left) или (i, right) -> (i+1, left)
        S4: переход вниз по левой полосе и остановка
        S5: переход вправо и остановка
        S6: переход вниз по правой полосе и остановка
        S7: переход влево и остановка
        """
        if prev_pos is None:
            return None
        
        prev_pair_idx, prev_side = prev_pos
        curr_pair_idx, curr_side = curr_pos
        
        if is_stopped:
            if prev_side == 'left' and curr_side == 'left' and curr_pair_idx == prev_pair_idx + 1:
                return "S4"
            elif prev_side == 'left' and curr_side == 'right' and curr_pair_idx == prev_pair_idx:
                return "S5"
            elif prev_side == 'right' and curr_side == 'right' and curr_pair_idx == prev_pair_idx + 1:
                return "S6"
            elif prev_side == 'right' and curr_side == 'left' and curr_pair_idx == prev_pair_idx:
                return "S7"
        else:
            if prev_side == 'left' and curr_side == 'left' and curr_pair_idx == prev_pair_idx + 1:
                return "S0"
            elif prev_side == 'left' and curr_side == 'right':
                if (curr_pair_idx == prev_pair_idx + 1) or (curr_pair_idx == prev_pair_idx):
                    return "S1"
                elif curr_pair_idx == prev_pair_idx and next_pos and next_pos[0] == curr_pair_idx + 1 and next_pos[1] == 'right':
                    return "S1"
            elif prev_side == 'right' and curr_side == 'right' and curr_pair_idx == prev_pair_idx + 1:
                return "S2"
            elif prev_side == 'right' and curr_side == 'left':
                if (curr_pair_idx == prev_pair_idx + 1) or (curr_pair_idx == prev_pair_idx):
                    return "S3"
                elif curr_pair_idx == prev_pair_idx and next_pos and next_pos[0] == curr_pair_idx + 1 and next_pos[1] == 'left':
                    return "S3"
        
        return "S0"
    
    def _extract_observation(self, item, action):
        ci, look, rel = item
        if action == 'purchase':
            return 'purchase'
        elif action == 'break':
            return 'break'
        elif look == 1:
            return 'click'
        else:
            return 'visit'
    
    def fit(self, chains):
        self.pi = initial_pi.copy()
        self.A = np.ones((self.n_states, self.n_states))
        self.B = np.ones((self.n_states, self.n_obs))
        
        state_counts = Counter()
        transition_counts = {}
        emission_counts = {}
        for i in range(self.n_states):
            transition_counts[i] = Counter()
            emission_counts[i] = Counter()
        
        state_sequences = []
        for chain in chains:
            if not chain or len(chain) < 2:
                continue
            
            prev_pos = None
            state_seq = []
            for i in range(len(chain)):
                pair_idx, side, item, action = chain[i]
                curr_pos = (pair_idx, side)
                
                next_pos = None
                if i < len(chain) - 1:
                    next_pair_idx, next_side, _, _ = chain[i + 1]
                    next_pos = (next_pair_idx, next_side)
                
                is_stopped = (action == 'purchase' or action == 'break')
                
                if prev_pos is not None:
                    state = self._determine_transition_state(prev_pos, curr_pos, next_pos, is_stopped)
                    if state:
                        state_idx = self.state_to_idx[state]
                        state_seq.append(state_idx)
                        
                        obs = self._extract_observation(item, action)
                        obs_idx = self.obs_to_idx[obs]
                        emission_counts[state_idx][obs_idx] += 1
                        
                        if is_stopped:
                            dead_state_idx = self.state_to_idx["DEAD"]
                            emission_counts[dead_state_idx][obs_idx] += 1
                else:
                    if i == 0:
                        first_state_idx = np.random.choice(4, p=self.pi[:4] / self.pi[:4].sum())
                        state_counts[first_state_idx] += 1
                
                prev_pos = curr_pos
            
            if len(state_seq) > 1:
                state_sequences.append(state_seq)
        
        for state_seq in state_sequences:
            for i in range(len(state_seq) - 1):
                from_idx = state_seq[i]
                to_idx = state_seq[i + 1]
                transition_counts[from_idx][to_idx] += 1
        
        total_initial = sum(state_counts.values())
        if total_initial > 0:
            self.pi = np.zeros(self.n_states)
            for state_idx, count in state_counts.items():
                if state_idx < 4:
                    self.pi[state_idx] = count / total_initial
            
            if self.pi[:4].sum() > 0:
                self.pi[:4] = self.pi[:4] / self.pi[:4].sum()
            else:
                self.pi = initial_pi.copy()
        else:
            self.pi = initial_pi.copy()
        
        for state_idx in range(self.n_states):
            total_transitions = sum(transition_counts[state_idx].values())
            if total_transitions > 0:
                for to_idx, count in transition_counts[state_idx].items():
                    self.A[state_idx, to_idx] = count
                self.A[state_idx] = self.A[state_idx] / (self.A[state_idx].sum() + 1e-10)
            else:
                self.A[state_idx] = P[state_idx].copy()
            
            total_emissions = sum(emission_counts[state_idx].values())
            if total_emissions > 0:
                for obs_idx, count in emission_counts[state_idx].items():
                    self.B[state_idx, obs_idx] = count
                self.B[state_idx] = self.B[state_idx] / (self.B[state_idx].sum() + 1e-10)
            else:
                self.B[state_idx] = self.B[state_idx] / self.B[state_idx].sum()
    
    def viterbi(self, observations):
        if not observations:
            return []
        
        T = len(observations)
        obs_indices = [self.obs_to_idx.get(obs, 0) for obs in observations]
        
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.pi * self.B[:, obs_indices[0]]
        delta[0][4:] = 0
        if delta[0][:4].sum() > 0:
            delta[0][:4] = delta[0][:4] / delta[0][:4].sum()
        else:
            delta[0][:4] = self.pi[:4] / self.pi[:4].sum()
        
        for t in range(1, T):
            for j in range(self.n_states):
                probs = delta[t-1] * self.A[:, j] * self.B[j, obs_indices[t]]
                delta[t, j] = probs.max()
                psi[t, j] = probs.argmax()
        
        path = np.zeros(T, dtype=int)
        path[T-1] = delta[T-1].argmax()
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return [self.idx_to_state[idx] for idx in path]
    
    def forward(self, observations):
        if not observations:
            return 0.0
        
        T = len(observations)
        obs_indices = [self.obs_to_idx.get(obs, 0) for obs in observations]
        
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.pi * self.B[:, obs_indices[0]]
        alpha[0] = alpha[0] / (alpha[0].sum() + 1e-10)
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = (alpha[t-1] * self.A[:, j]).sum() * self.B[j, obs_indices[t]]
            alpha[t] = alpha[t] / (alpha[t].sum() + 1e-10)
        
        return alpha[T-1].sum()
    
    def generate_sequence(self, max_length=20):
        sequence = []
        states = []
        
        current_state_idx = np.random.choice(4, p=self.pi[:4] / self.pi[:4].sum())
        states.append(self.idx_to_state[current_state_idx])
        
        for _ in range(max_length):
            obs_probs = self.B[current_state_idx]
            obs_idx = np.random.choice(self.n_obs, p=obs_probs)
            obs = self.idx_to_obs[obs_idx]
            sequence.append(obs)
            
            if obs == 'purchase' or obs == 'break':
                break
            
            next_state_probs = self.A[current_state_idx]
            current_state_idx = np.random.choice(self.n_states, p=next_state_probs)
            states.append(self.idx_to_state[current_state_idx])
            
            if self.idx_to_state[current_state_idx] == "DEAD":
                break
        
        return sequence, states
    
    def get_state_info(self):
        return {
            'n_states': self.n_states,
            'states': self.hidden_states,
            'observations': self.observations
        }

def prepare_training_data(chains, hmm_model):
    observations_sequences = []
    state_sequences = []
    
    for chain in chains:
        if not chain or len(chain) < 1:
            continue
            
        obs_seq = []
        state_seq = []
        
        if len(chain) < 2:
            continue
        
        prev_pos = None
        for i, (pair_idx, side, item, action) in enumerate(chain):
            curr_pos = (pair_idx, side)
            
            next_pos = None
            if i < len(chain) - 1:
                next_pair_idx, next_side, _, _ = chain[i + 1]
                next_pos = (next_pair_idx, next_side)
            
            is_stopped = (action == 'purchase' or action == 'break')
            
            ci, look, rel = item
            if action == 'purchase':
                obs = 'purchase'
            elif action == 'break':
                obs = 'break'
            elif look == 1:
                obs = 'click'
            else:
                obs = 'visit'
            
            if prev_pos is not None:
                state = hmm_model._determine_transition_state(prev_pos, curr_pos, next_pos, is_stopped)
                if state:
                    state_seq.append(state)
                    obs_seq.append(obs)
            
            prev_pos = curr_pos
        
        if obs_seq and state_seq and len(obs_seq) == len(state_seq):
            if len(state_seq) > 0 and state_seq[0] in ["S4", "S5", "S6", "S7"]:
                continue
            observations_sequences.append(obs_seq)
            state_sequences.append(state_seq)
    
    return observations_sequences, state_sequences

if __name__ == "__main__":
    feed, chains = generate_chains(n_chains=1000, n_pairs=100, p_look=0.3, p_break=0.15)
    
    print(f"Generated {len(chains)} chains")
    print(f"Average chain length: {np.mean([len(c) for c in chains]):.2f}")
    
    hmm = UserBehaviorHMM(n_pairs=1000)
    hmm.fit(chains)
    
    obs_sequences, state_sequences = prepare_training_data(chains, hmm)
    
    info = hmm.get_state_info()
    print(f"\nHMM Model Info:")
    print(f"  Number of hidden states: {info['n_states']}")
    print(f"  States: {info['states']}")
    print(f"  Number of observations: {len(info['observations'])}")
    print(f"  Observations: {info['observations']}")
    
    print(f"\nInitial probabilities:")
    for idx, state in enumerate(hmm.hidden_states):
        if hmm.pi[idx] > 0:
            print(f"  {state}: {hmm.pi[idx]:.4f}")
    
    print(f"\nTesting Viterbi on all chains:")
    accuracies = []
    
    for i, (test_obs, actual_states) in enumerate(zip(obs_sequences, state_sequences)):
        predicted_states = hmm.viterbi(test_obs)
        
        min_len = min(len(actual_states), len(predicted_states))
        if min_len > 0:
            accuracy = sum(1 for a, p in zip(actual_states[:min_len], predicted_states[:min_len]) if a == p) / min_len
            accuracies.append(accuracy)
    
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print(f"  Average accuracy: {avg_accuracy:.2%} ± {std_accuracy:.2%}")
        print(f"  Min accuracy: {min(accuracies):.2%}")
        print(f"  Max accuracy: {max(accuracies):.2%}")
        print(f"  Number of chains: {len(accuracies)}")
        
        print(f"\nExample (first chain):")
        test_obs = obs_sequences[0]
        predicted_states = hmm.viterbi(test_obs)
        actual_states = state_sequences[0]
        print(f"  Observations: {test_obs}")
        print(f"  Actual states: {actual_states}")
        print(f"  Predicted states: {predicted_states}")
        if len(actual_states) > 0 and len(predicted_states) > 0:
            min_len = min(len(actual_states), len(predicted_states))
            accuracy = sum(1 for a, p in zip(actual_states[:min_len], predicted_states[:min_len]) if a == p) / min_len
            print(f"  Accuracy: {accuracy:.2%}")
    
    print(f"\nGenerating new sequence:")
    gen_obs, gen_states = hmm.generate_sequence(max_length=10)
    print(f"  Observations: {gen_obs}")
    print(f"  States: {gen_states}")