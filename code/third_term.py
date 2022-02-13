#!/usr/bin/env python

# testing cancellation of third term after substitution of μ

# μ = sum_{l=1}^K T_{-lc}(y_l)
#
# T_{kc}(μ) = \sum_{l=1}^K y_{l,(k-l)c + n % N}

import matplotlib.pyplot as plt
import numpy as np

def compute_first(*, K, c, N, y_list):
    result = 0
    for k in range(0, K):
        for n in range(0, N):
            result += y_list[k][n]
    return result

def compute_second(*, K, c, N, y_list):
    result = 0
    for k in range(0, K):
        for n in range(0, N):
            for l in range(0, K):
                 # += y_list[l][((k - l) * c + n) % N]
                result += y_list[k][n] * y_list[l][((k - l) * c + N) % N]

    return result

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

def compute_multiml(*, K, c, N, y_list):
    Y_list = [np.fft.fft(y) for y in y_list]

    cg_list = []
    for diff in range(0, len(y_list)):
        cg = np.zeros(len(y_list[0]))
        # print(f'diff -- {diff}')
        for first in range(0, len(y_list) - diff):
            # print(f'({first}, {first + diff})')
            cg += np.fft.ifft(Y_list[first] * Y_list[first + diff].conj()).real

        cg_list.append(cg)

    result = np.zeros(len(y_list[0]))
    for diff, cg in enumerate(cg_list, 1):
        downsampled = cg[::diff]
        result[:len(downsampled)] += downsampled

    return result


def compute_combined(*, K, c, N, y_list):
    result = 0
    for k in range(0, K):
        for n in range(0, N):
            temp = 0
            for l in range(0, K):
                temp += y_list[l][((k - l) * c + n) % N]

            result += (y_list[k][n] - temp)**2

    return result

def compute_indices(*, K, c, N, y_list):
    indices = np.zeros((K, N))
    for k in range(0, K):
        for n in range(0, N):
            temp = 0
            for l in range(0, K):
                indices[l][((k - l) * c + n) % N] += 1

    return indices


K = 10
N = 30
c = 1
# y = np.arange(N)
y = np.random.random(N)
y = np.array([1,2,3,4,4,4,4,5,6,3,2,1,3,3,1,1,3,3,1,1,3,3,1,1,3,3,1,1,3,3])
# y = np.zeros(N)
# y[0] = 1
# y[1] = 1
# y[2] = 1

y_list = [np.roll(y, c * k) for k in range(K)]


def scan(func, *, c, K, N, y_list):
    scanned = []
    for c in range(N):
        scanned.append(func(c=c, K=K, N=N, y_list=y_list))

    return scanned

# scanned_indices = []
# for c in range(N):
#     scanned_indices.append(compute_indices(c=c, K=K, N=N, y_list=y_list))

def norm(x):
    if not type(x) is np.ndarray:
        x = np.array(x)
    return x
    return (x - x.min()) / (x.max() - x.min())

first = norm(scan(compute_first, c=c, K=K, N=N, y_list=y_list))
second = norm(scan(compute_second, c=c, K=K, N=N, y_list=y_list))
third = norm(scan(compute_third, c=c, K=K, N=N, y_list=y_list))
combined = norm(scan(compute_combined, c=c, K=K, N=N, y_list=y_list))
added = -2 * second + third
multiml = 30 * norm(compute_multiml(c=c, K=K, N=N, y_list=y_list))

plt.plot(first, label='first')
plt.plot(second, label='second')
plt.plot(third, label='third')
# plt.plot(combined, '*', label='combined')
plt.plot(added, 'o', label='added')
plt.plot(multiml, label='multiml')

plt.legend(loc='upper right')
plt.show()
