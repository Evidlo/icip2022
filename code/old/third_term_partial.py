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
# y = [1, 2, 3, 4]

y_list = [np.roll(y, c * k) for k in range(K)]

# %% third --------------------

def compute_third(*, K, c, N, y_list):
    result = 0
    for k in range(0, K):
        for n in range(0, N):
            temp = 0
            for l in range(0, K):
                temp += y_list[l][((k - l) * c + n) % N]
            result += temp**2
            # result += temp

    return result

# removed self mults and extra K loop
def compute_third2(*, K, c, N, y_list):
    result = 0
    # get rid of K loop
    for k in range(0, 1):
        for n in range(0, N):
            temp = 0
            for l in range(0, K):
                temp += y_list[l][((k - l) * c + n) % N]
            temp = temp**2
            # print(temp, end='')
            # subtract self mults
            for l in range(0, K):
                temp -= y_list[l][((k - l) * c + n) % N]**2
            # remove doubles
            temp /= 2
            result += temp
            # print(' -', temp)

    return result

print(compute_third(K=K, c=0, N=N, y_list=y_list))
print(compute_third2(K=K, c=0, N=N, y_list=y_list))
print()
print(compute_third(K=K, c=1, N=N, y_list=y_list))
print(compute_third2(K=K, c=1, N=N, y_list=y_list))
print()
print(compute_third(K=K, c=2, N=N, y_list=y_list))
print(compute_third2(K=K, c=2, N=N, y_list=y_list))
print()
print(compute_third(K=K, c=3, N=N, y_list=y_list))
print(compute_third2(K=K, c=3, N=N, y_list=y_list))

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
