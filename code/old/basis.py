#!/usr/bin/env python


import numpy as np
from itertools import product, combinations

N = 4
cols = 4
outputs = N*(N-1)//2
col_vals = np.arange(cols)

M = np.zeros((outputs, cols))

vecs = np.array(list(product(*[[0, 1]]*outputs)))

def correlations(vec):

for m_cols in combinations(vecs, cols):
    M = np.vstack(m_cols).T
    orig_rank = np.linalg.matrix_rank(
    )
    aug_rank =
