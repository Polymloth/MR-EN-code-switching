import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

freqs = []
with open('Multilingual_only/MULTILINGUAL-ONLY-en-types.tsv', encoding='utf-8') as f:
    for line in f:
        token_freq = line.rstrip('\n').split('\t')
        freqs.append(int(token_freq[1]))

counts = np.array(freqs)

#frequency-of-frequencies f(n)
unique_n, type_count = np.unique(counts, return_counts=True)
emp_f = type_count / type_count.sum()

min_types_per_n = 5
n_min, n_max = 3, 2000

mask_fit = (
    (type_count >= min_types_per_n) &
    (unique_n >= n_min) &
    (unique_n <= n_max)
)

x = np.log10(unique_n[mask_fit].astype(float))
y = np.log10(emp_f[mask_fit].astype(float))
slope, intercept, r, _, _ = linregress(x, y)
emp_beta = -slope

n_plot = unique_n.astype(float)
expcd_f = (n_plot ** -2)
expcd_f /= expcd_f.sum()

f_fit = (10 ** intercept) * (n_plot ** slope)

print(f'Empirical beta =(approx) {emp_beta:.3f}')

plt.figure(figsize=(7,5))
plt.loglog(unique_n, emp_f, 'o', alpha=0.7, label=r'Empirical $f(n)$')
plt.loglog(n_plot, expcd_f, '-', label=r'Theoretical $1/n^2$')
plt.loglog(n_plot, f_fit, '--', label=rf'Empirical fit $1/n^{{{emp_beta:.2f}}}$')
plt.xlabel('n (token count per type)')
plt.ylabel('Probability f(n)')
plt.legend()
plt.tight_layout()
plt.show()
# yay looks good :)

np.savetxt('empirical_type-freq_data.dat', np.column_stack((unique_n, emp_f)), fmt='%.6e', delimiter='\t', header='n\tf_n', comments='')