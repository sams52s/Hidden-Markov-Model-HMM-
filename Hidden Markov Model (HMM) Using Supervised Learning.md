# Hidden Markov Model (HMM) Using Supervised Learning

This section explains the implementation of a Hidden Markov Model (HMM) using a supervised learning approach. Below is a step-by-step breakdown of the process.

## 1. Define the HMM Components

The HMM consists of several key elements:

- **States (`S`)**:
  - The set of hidden states in the HMM (e.g., `Rainy`, `Sunny`).
  - Number of states: `N`.
  
- **Observations (`O`)**:
  - The set of possible observable symbols (e.g., `walk`, `shop`, `clean`).
  - Number of observations: `M`.
  
- **Transition Probabilities (`A`)**:
  - Probabilities of transitioning from one hidden state to another.
  - `A[i][j]` represents the probability of transitioning from state `i` to state `j`.

- **Emission Probabilities (`B`)**:
  - Probabilities of observing a particular observation given a state.
  - `B[i][k]` represents the probability of observing symbol `k` when in state `i`.

- **Initial Probabilities (`π`)**:
  - The probability of starting in each state.
  - `π[i]` represents the probability of starting in state `i`.

## 2. Collect Training Data

- Use labeled sequences of observations and corresponding hidden states.
- Each training sequence includes:
  - **Observation sequence**: A sequence of observable symbols (e.g., `['walk', 'shop', 'walk']`).
  - **State sequence**: A sequence of hidden states corresponding to each observation (e.g., `['Rainy', 'Sunny', 'Rainy']`).

## 3. Estimate Parameters

Based on the labeled data, we estimate the HMM parameters: transition probabilities, emission probabilities, and initial probabilities.

1. **Transition Probabilities (`A`)**:
   - Count transitions between states.
   - For each state `i` and state `j`:
     \[
     A[i][j] = \frac{\text{Count of transitions from state } i \text{ to state } j}{\text{Total transitions from state } i}
     \]

2. **Emission Probabilities (`B`)**:
   - Count occurrences of each observation for every state.
   - For each state `i` and observation `k`:
     \[
     B[i][k] = \frac{\text{Count of observation } k \text{ in state } i}{\text{Total occurrences of state } i}
     \]

3. **Initial Probabilities (`π`)**:
   - Count occurrences of each state as the starting state.
   - For each state `i`:
     \[
     \pi[i] = \frac{\text{Number of sequences starting in state } i}{\text{Total number of sequences}}
     \]

## 4. Code Implementation

Here’s the Python code to implement the supervised learning for HMM:

```python
states = ['Rainy', 'Sunny']
observations = ['walk', 'shop', 'clean']

training_data = [
    ('Rainy', 'walk'),
    ('Rainy', 'shop'),
    ('Sunny', 'walk'),
    ('Sunny', 'clean'),
    ('Rainy', 'clean')
]

transition_counts = {state: {s: 0 for s in states} for state in states}
emission_counts = {state: {obs: 0 for obs in observations} for state in states}
initial_counts = {state: 0 for state in states}

for i, (state, obs) in enumerate(training_data):
    if i == 0:
        initial_counts[state] += 1
    else:
        prev_state = training_data[i-1][0]
        transition_counts[prev_state][state] += 1
    emission_counts[state][obs] += 1

transition_probs = {state: {s: count / sum(transition_counts[state].values()) for s, count in trans.items()} for state, trans in transition_counts.items()}
emission_probs = {state: {obs: count / sum(emission_counts[state].values()) for obs, count in emis.items()} for state, emis in emission_counts.items()}
initial_probs = {state: count / sum(initial_counts.values()) for state, count in initial_counts.items()}

print("Transition Probabilities:", transition_probs)
print("Emission Probabilities:", emission_probs)
print("Initial Probabilities:", initial_probs)


```

## 5. Training Process Overview
    - Step 1: Initialize count matrices for transitions, emissions, and initial probabilities.
    - Step 2: Convert sequences to indices and update counts for transitions, emissions, and initial states.
    - Step 3: Normalize the counts to obtain probabilities for transition (A), emission (B), and initial states (π).

## Result:
```
Transition Probabilities: {'Rainy': {'Rainy': 0.5, 'Sunny': 0.5}, 'Sunny': {'Rainy': 0.5, 'Sunny': 0.5}}
Emission Probabilities: {'Rainy': {'walk': 0.3333333333333333, 'shop': 0.3333333333333333, 'clean': 0.3333333333333333}, 'Sunny': {'walk': 0.5, 'shop': 0.0, 'clean': 0.5}}
Initial Probabilities: {'Rainy': 1.0, 'Sunny': 0.0}

=== Code Execution Successful ===

``` 