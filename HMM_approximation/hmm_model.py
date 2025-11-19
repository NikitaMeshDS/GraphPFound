import numpy as np
from hmm_matrix import P, pi as initial_pi

class UserBehaviorHMM:
    """HMM для моделирования поведения пользователя: скрытые состояния - перемещения, наблюдения - действия."""
    
    def __init__(self, n_pairs):
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
    
    def _extract_observation(self, item, action):
        """Преобразует (item, action) в наблюдение: visit/click/purchase/break."""
        ci, look, rel = item
        if action == 'purchase':
            return 'purchase'
        elif action == 'break':
            return 'break'
        elif look == 1:
            return 'click'
        else:
            return 'visit'
    
    def _forward_backward(self, observations):
        """E-step алгоритма Баум-Велч: вычисляет gamma и xi через forward-backward."""
        T = len(observations)
        obs_indices = [self.obs_to_idx.get(obs, 0) for obs in observations]
        
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        
        alpha[0] = self.pi * self.B[:, obs_indices[0]]
        alpha[0][8:] = 0
        if alpha[0][:8].sum() > 0:
            alpha[0][:8] = alpha[0][:8] / alpha[0][:8].sum()
        else:
            alpha[0][:8] = self.pi[:8] / self.pi[:8].sum()
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = (alpha[t-1] * self.A[:, j]).sum() * self.B[j, obs_indices[t]]
            if alpha[t].sum() > 0:
                alpha[t] = alpha[t] / alpha[t].sum()
        
        ends_with_stop = observations[-1] in ['purchase', 'break']
        if ends_with_stop:
            beta[T-1, :8] = 1.0
            beta[T-1, 8] = 0.0
        else:
            beta[T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = (self.A[i, :] * self.B[:, obs_indices[t+1]] * beta[t+1, :]).sum()
            if beta[t].sum() > 0:
                beta[t] = beta[t] / beta[t].sum()
        
        gamma = alpha * beta
        for t in range(T):
            if gamma[t].sum() > 0:
                gamma[t] = gamma[t] / gamma[t].sum()
        
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    if alpha[t, i] > 0:
                        xi[t, i, j] = alpha[t, i] * self.A[i, j] * self.B[j, obs_indices[t+1]] * beta[t+1, j]
            if xi[t].sum() > 0:
                xi[t] = xi[t] / xi[t].sum()
        
        return gamma, xi
    
    def fit(self, chains, max_iter=50, tol=1e-4):
        """Обучение HMM алгоритмом Баум-Велч на основе только наблюдений (действий)."""
        obs_sequences = []
        for chain in chains:
            if not chain or len(chain) < 2:
                continue
            obs_seq = []
            for chain_item in chain:
                if len(chain_item) >= 4:
                    _, _, item, action = chain_item[:4]
                    obs_seq.append(self._extract_observation(item, action))
            if len(obs_seq) >= 2:
                obs_sequences.append(obs_seq)
        
        if not obs_sequences:
            return
        
        self.pi = initial_pi.copy()
        self.A = P.copy()
        
        # Инициализация матрицы эмиссий B: S0-S3 чаще visit/click, S4-S7 чаще purchase/break
        self.B = np.ones((self.n_states, self.n_obs)) / self.n_obs
        for i in range(8):
            if i < 4:
                self.B[i] = np.array([0.7, 0.2, 0.05, 0.05])
            else:
                self.B[i] = np.array([0.1, 0.1, 0.4, 0.4])
        self.B[8] = np.array([0.0, 0.0, 0.5, 0.5])
        
        for iteration in range(max_iter):
            old_A, old_B = self.A.copy(), self.B.copy()
            
            gamma_sum = np.zeros(self.n_states)
            xi_sum = np.zeros((self.n_states, self.n_states))
            emission_sum = np.zeros((self.n_states, self.n_obs))
            
            for obs_seq in obs_sequences:
                if len(obs_seq) < 2:
                    continue
                
                gamma, xi = self._forward_backward(obs_seq)
                gamma_sum += gamma.sum(axis=0)
                
                for t in range(len(obs_seq) - 1):
                    xi_sum += xi[t]
                
                obs_indices = [self.obs_to_idx.get(obs, 0) for obs in obs_seq]
                for t in range(len(obs_seq)):
                    emission_sum[:, obs_indices[t]] += gamma[t]
            
            if gamma_sum[:8].sum() > 0:
                self.pi[:8] = gamma_sum[:8] / gamma_sum[:8].sum()
                self.pi[8] = 0.0
            else:
                self.pi = initial_pi.copy()
            
            # M-step: переоценка параметров
            for i in range(self.n_states):
                if gamma_sum[i] > 0:
                    self.A[i] = xi_sum[i] / (xi_sum[i].sum() + 1e-10)
                    self.B[i] = emission_sum[i] / (emission_sum[i].sum() + 1e-10)
                else:
                    self.A[i] = P[i].copy()
                    self.B[i] = np.ones(self.n_obs) / self.n_obs
            
            # S4-S7 должны переходить в DEAD, используем априорную матрицу если нет данных
            for i in [4, 5, 6, 7]:
                if xi_sum[i].sum() == 0:
                    self.A[i] = P[i].copy()
            
            if np.abs(self.A - old_A).max() < tol and np.abs(self.B - old_B).max() < tol:
                break
    
    def viterbi(self, observations):
        """Алгоритм Витерби: находит наиболее вероятную последовательность скрытых состояний."""
        if not observations:
            return []
        
        T = len(observations)
        obs_indices = [self.obs_to_idx.get(obs, 0) for obs in observations]
        ends_with_stop = observations[-1] in ['purchase', 'break']
        
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.pi * self.B[:, obs_indices[0]]
        delta[0][8:] = 0
        if delta[0][:8].sum() > 0:
            delta[0][:8] = delta[0][:8] / delta[0][:8].sum()
        else:
            delta[0][:8] = self.pi[:8] / self.pi[:8].sum()
        
        for t in range(1, T):
            for j in range(self.n_states):
                probs = delta[t-1] * self.A[:, j] * self.B[j, obs_indices[t]]
                delta[t, j] = probs.max()
                psi[t, j] = probs.argmax()
            
            if t == T - 1 and ends_with_stop:
                delta[t, 8] = 0
        
        path = np.zeros(T, dtype=int)
        path[T-1] = delta[T-1, :8].argmax() if ends_with_stop else delta[T-1].argmax()
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return [self.idx_to_state[idx] for idx in path]
    
    def generate_sequence(self, max_length=20):
        """Генерирует последовательность наблюдений и состояний."""
        sequence, states = [], []
        current_state_idx = np.random.choice(8, p=self.pi[:8] / self.pi[:8].sum())
        
        for _ in range(max_length):
            obs_probs = self.B[current_state_idx]
            obs_idx = np.random.choice(self.n_obs, p=obs_probs)
            obs = self.idx_to_obs[obs_idx]
            sequence.append(obs)
            states.append(self.idx_to_state[current_state_idx])
            
            if obs == 'purchase' or obs == 'break':
                break
            
            next_state_probs = self.A[current_state_idx]
            current_state_idx = np.random.choice(self.n_states, p=next_state_probs)
            
            if self.idx_to_state[current_state_idx] == "DEAD":
                break
        
        return sequence, states

def prepare_training_data(chains, hmm_model):
    """Извлекает последовательности наблюдений и состояний из цепочек."""
    observations_sequences = []
    state_sequences = []
    
    for chain in chains:
        if not chain or len(chain) < 2:
            continue
        
        obs_seq = []
        state_seq = []
        
        for chain_item in chain:
            if len(chain_item) >= 4:
                _, _, item, action = chain_item[:4]
                obs_seq.append(hmm_model._extract_observation(item, action))
                
                if len(chain_item) == 5:
                    state = chain_item[4]
                    if state and state != "DEAD":
                        state_seq.append(state)
        
        if obs_seq and len(state_seq) == len(obs_seq):
            observations_sequences.append(obs_seq)
            state_sequences.append(state_seq)
        elif obs_seq:
            observations_sequences.append(obs_seq)
            state_sequences.append(None)
    
    return observations_sequences, state_sequences