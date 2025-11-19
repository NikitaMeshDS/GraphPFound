import numpy as np
from hmm_model import UserBehaviorHMM, prepare_training_data
from generate_chains import generate_chains

if __name__ == "__main__":
    feed, chains = generate_chains(n_chains=1000, n_pairs=100, p_look=0.3, p_break=0.15)
    
    print(f"Generated {len(chains)} chains")
    print(f"Average chain length: {np.mean([len(c) for c in chains]):.2f}")
    
    hmm = UserBehaviorHMM(n_pairs=100)
    hmm.fit(chains)
    
    obs_sequences, state_sequences = prepare_training_data(chains, hmm)
    
    print(f"\nHMM Model Info:")
    print(f"  States: {hmm.hidden_states}")
    print(f"  Observations: {hmm.observations}")
    
    print(f"\nInitial probabilities:")
    for idx, state in enumerate(hmm.hidden_states):
        if hmm.pi[idx] > 0:
            print(f"  {state}: {hmm.pi[idx]:.4f}")
    
    print(f"\nTransition matrix A:")
    print("  " + " ".join([f"{s:>6}" for s in hmm.hidden_states]))
    for i, state in enumerate(hmm.hidden_states):
        row_str = " ".join([f"{hmm.A[i, j]:6.3f}" for j in range(hmm.n_states)])
        print(f"  {state} {row_str}")
    
    print(f"\nTesting Viterbi on all chains:")
    accuracies = []
    
    for test_obs, actual_states in zip(obs_sequences, state_sequences):
        if actual_states is None:
            continue
        predicted_states = hmm.viterbi(test_obs)
        
        min_len = min(len(actual_states), len(predicted_states))
        if min_len > 0:
            accuracy = sum(1 for a, p in zip(actual_states[:min_len], predicted_states[:min_len]) if a == p) / min_len
            accuracies.append(accuracy)
    
    if accuracies:
        print(f"  Average accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
        print(f"  Min accuracy: {min(accuracies):.2%}")
        print(f"  Max accuracy: {max(accuracies):.2%}")
        print(f"  Number of chains: {len(accuracies)}")
        
        print(f"\nExample (first chain):")
        test_obs = obs_sequences[0]
        predicted_states = hmm.viterbi(test_obs)
        actual_states = state_sequences[0]
        print(f"  Observations: {test_obs}")
        if actual_states is not None:
            print(f"  Actual states: {actual_states}")
        print(f"  Predicted states: {predicted_states}")
        if actual_states is not None and len(actual_states) > 0 and len(predicted_states) > 0:
            min_len = min(len(actual_states), len(predicted_states))
            accuracy = sum(1 for a, p in zip(actual_states[:min_len], predicted_states[:min_len]) if a == p) / min_len
            print(f"  Accuracy: {accuracy:.2%}")
    
    print(f"\nGenerating new sequence:")
    gen_obs, gen_states = hmm.generate_sequence(max_length=10)
    print(f"  Observations: {gen_obs}")
    print(f"  States: {gen_states}")

