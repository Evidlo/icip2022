#!/usr/bin/env python

# testing cancellation of third term after substitution of μ

# μ = sum_{l=1}^K T_{-lc}(y_l)
#
# T_{kc}(μ) = \sum_{l=1}^K y_{l,(k-l)c + n % N}

import matplotlib.pyplot as plt
import numpy as np

N = 4
K = 3
c = 1

y = np.zeros(N)
y[0] = 1

y_list = [np.roll(y, c * k) for k in range(K)]

# %% second --------------------

def compute_second(*, K, c, N, y_list):
    result = 0
    for n in range(0, N):
        for k in range(0, K):
            for l in range(0, K):
                 # += y_list[l][((k - l) * c + n) % N]
                val = y_list[k][n] * y_list[l][((k - l) * c + n) % N]
                # print((k, n), (l, ((k-l) * c + n) % N), val)
                result += val
        # print()

    return result

print(compute_second(K=K, c=0, N=N, y_list=y_list))
print()
print(compute_second(K=K, c=1, N=N, y_list=y_list))
print()
print(compute_second(K=K, c=2, N=N, y_list=y_list))
print()
print(compute_second(K=K, c=3, N=N, y_list=y_list))

# %% multiml ------------------

Y_list = [np.fft.fft(y) for y in y_list]

cg_list = []
for diff in range(0, len(y_list)):
    cg = np.zeros(len(y_list[0]))
    # print(f'diff -- {diff}')
    for first in range(0, len(y_list) - diff):
        # print(f'({first}, {first + diff})')
        cg += np.fft.ifft(Y_list[first] * Y_list[first + diff].conj()).real

    cg_list.append(cg)
