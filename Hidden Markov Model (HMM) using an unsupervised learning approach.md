## Hidden Markov Model (HMM) Using Baum-Welch Algorithm

This section explains the implementation of a Hidden Markov Model (HMM) using an unsupervised learning approach with the Baum-Welch algorithm.

### 1. Define the HMM Components
- **States**: The hidden states in the model (e.g., `Rainy`, `Sunny`).
- **Observations**: The observed events or symbols (e.g., `walk`, `shop`, `clean`).
- **Transition Probabilities**: The probabilities of transitioning between hidden states.
- **Emission Probabilities**: The probabilities of observing a particular symbol given a hidden state.
- **Initial Probabilities**: The probabilities of starting in each hidden state.

### 2. Initialize Parameters
We randomly initialize the following probabilities:
- **Transition Probabilities**: Random values representing state transitions.
- **Emission Probabilities**: Random values representing observations given a state.
- **Initial Probabilities**: Random values representing the likelihood of starting in each state.

### 3. Baum-Welch Algorithm
The Baum-Welch algorithm, an instance of the Expectation-Maximization (EM) algorithm, is used to estimate the unknown parameters of the HMM. It iteratively alternates between two steps:
- **E-step (Expectation)**: Computes the probabilities of the hidden states using the current model parameters.
- **M-step (Maximization)**: Updates the model parameters (transition, emission, and initial probabilities) to maximize the likelihood of the observed sequence.

### 4. Run the Algorithm
Here’s the Python code to implement the supervised learning for HMM:

```python
import numpy as np

states = ['Rainy', 'Sunny']
observations = ['walk', 'shop', 'clean']
obs_seq = ['walk', 'shop', 'walk', 'clean', 'clean']


num_states = len(states)
num_obs = len(observations)

transition_probs = np.random.rand(num_states, num_states)
transition_probs /= transition_probs.sum(axis=1, keepdims=True)
emission_probs = np.random.rand(num_states, num_obs)
emission_probs /= emission_probs.sum(axis=1, keepdims=True)


initial_probs = np.random.rand(num_states)
initial_probs /= initial_probs.sum()

state_map = {state: i for i, state in enumerate(states)}
obs_map = {obs: i for i, obs in enumerate(observations)}

obs_seq_idx = [obs_map[obs] for obs in obs_seq]
def baum_welch(obs_seq, num_states, num_obs, transition_probs, emission_probs, initial_probs, max_iter=100):
    T = len(obs_seq)
    for _ in range(max_iter):
        alpha = np.zeros((T, num_states))
        beta = np.zeros((T, num_states))
        gamma = np.zeros((T, num_states))
        xi = np.zeros((T-1, num_states, num_states))
        alpha[0] = initial_probs * emission_probs[:, obs_seq[0]]
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ transition_probs) * emission_probs[:, obs_seq[t]]
        beta[-1] = 1
        for t in range(T-2, -1, -1):
            beta[t] = (beta[t+1] * emission_probs[:, obs_seq[t+1]]) @ transition_probs.T
        for t in range(T):
            gamma[t] = alpha[t] * beta[t] / (alpha[t] * beta[t]).sum()
        for t in range(T-1):
            xi[t] = (alpha[t][:, np.newaxis] * transition_probs * emission_probs[:, obs_seq[t+1]] * beta[t+1]) / (alpha[t] * beta[t]).sum()
        initial_probs = gamma[0]
        transition_probs = xi.sum(axis=0) / gamma[:-1].sum(axis=0)[:, np.newaxis]
        
        for k in range(num_obs):
            mask = np.array(obs_seq) == k 
            emission_probs[:, k] = gamma[mask].sum(axis=0) / gamma.sum(axis=0)

    return transition_probs, emission_probs, initial_probs

transition_probs, emission_probs, initial_probs = baum_welch(obs_seq_idx, num_states, num_obs, transition_probs, emission_probs, initial_probs)

print("Transition Probabilities:", transition_probs)
print("Emission Probabilities:", emission_probs)
print("Initial Probabilities:", initial_probs)
```

## Result:
``` 
Transition Probabilities: [[0.5 0.5]
 [1.  0. ]]
Emission Probabilities: [[3.26148267e-76 3.33333333e-01 6.66666667e-01]
 [1.00000000e+00 0.00000000e+00 1.05960995e-14]]
Initial Probabilities: [0. 1.]

=== Code Execution Successful ===
```
This section demonstrates how to implement the Baum-Welch algorithm to learn the parameters of a Hidden Markov Model (HMM) using unsupervised learning.
