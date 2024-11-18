import numpy as np

# Constants
m = 1000
b_values = [0.01, 0.005, 0.001]
step_size = m // 8

# P_false
def compute_false_positive_prob(n, m, k):
    return (1 - np.exp(-k * m / n)) ** k

optimal_results = {}

for b in b_values:

    n = step_size
    while (n <= 20 * m):
        k_opt = int(round(n / m * np.log(2)))
        p_false_positive = compute_false_positive_prob(n, m, k_opt)
        
        if p_false_positive <= b:
            optimal_results[b] = {
                'n': n,
                'k': k_opt,
                'P(FPP)': p_false_positive
            }
            break
        n += step_size

print(optimal_results)