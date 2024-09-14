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
