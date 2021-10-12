# Tested
# WMF https://github.com/benanne/wmf
# SparseMF https://github.com/jeh0753/sparseMF
# NMF https://github.com/kimjingu/nonnegfac-python
# Implicit https://github.com/benfred/implicit
# + Many SVD algorithms

# Not tested
# modl(fMRIDictFact) https://github.com/arthurmensch/modl , not compatible with latest scikitlearn dependency
# Recoder https://github.com/amoussawi/recoder , errors thrown from CUDA
# NetSMF https://github.com/xptree/NetSMF , need lots of work for preparing input, does not work with numpy/scipy out of the box
# CuMF https://github.com/cuMF/ : I suggest you check these out, they might be well performant as they are written in C+ and CUDA directly. I can’t use them directly in python, some work is needed
# https://github.com/benedekrozemberczki/karateclub This is worth a look, it's a great graph embedding library. But I could not find an algo for your problem.


import sys
import os
import time
import itertools

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=20,15
import scipy.sparse
import numpy as np

# svd
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sparsesvd import sparsesvd
from gensim.models.lsimodel import stochastic_svd

# weighted matrix factorization, deep learning and more
import nonnegfac
from nonnegfac.nmf import NMF
from nonnegfac.nmf import NMF_ANLS_BLOCKPIVOT
from nonnegfac.nmf import NMF_ANLS_AS_NUMPY
from nonnegfac.nmf import NMF_ANLS_AS_GROUP
from nonnegfac.nmf import NMF_HALS
from nonnegfac.nmf import NMF_MU

# daal4py
import daal4py as d4p


# Code from https://github.com/benanne/wmf/blob/master/wmf.py
def linear_surplus_confidence_matrix(B, alpha):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * B
    S = B.copy()
    S.data = alpha * S.data
    return S

def log_surplus_confidence_matrix(B, alpha, epsilon):
    # To construct the surplus confidence matrix, we need to operate only on the nonzero elements.
    # This is not possible: S = alpha * np.log(1 + B / epsilon)
    S = B.copy()
    S.data = alpha * np.log(1 + S.data / epsilon)
    return S

def iter_rows(S):
    """
    Helper function to iterate quickly over the data and indices of the
    rows of the S matrix. A naive implementation using indexing
    on S is much, much slower.
    """
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        yield i, S.data[lo:hi], S.indices[lo:hi]

def recompute_factors(Y, S, lambda_reg, dtype='float32'):
    """
    recompute matrix X from Y.
    X = recompute_factors(Y, S, lambda_reg)
    This can also be used for the reverse operation as follows:
    Y = recompute_factors(X, ST, lambda_reg)

    The comments are in terms of X being the users and Y being the items.
    """
    m = S.shape[0] # m = number of users
    f = Y.shape[1] # f = number of factors
    YTY = np.dot(Y.T, Y) # precompute this
    YTYpI = YTY + lambda_reg * np.eye(f)
    X_new = np.zeros((m, f), dtype=dtype)

    for k, s_u, i_u in iter_rows(S):
        Y_u = Y[i_u] # exploit sparsity
        A = np.dot(s_u + 1, Y_u)
        YTSY = np.dot(Y_u.T, (Y_u * s_u.reshape(-1, 1)))
        B = YTSY + YTYpI

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv)
        X_new[k] = np.linalg.solve(B.T, A.T).T # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new

def recompute_factors_bias(Y, S, lambda_reg, dtype='float32'):
    """
    Like recompute_factors, but the last column of X and Y is treated as
    a bias vector.
    """
    m = S.shape[0] # m = number of users
    f = Y.shape[1] - 1 # f = number of factors

    b_y = Y[:, f] # vector of biases

    Y_e = Y.copy()
    Y_e[:, f] = 1 # factors with added column of ones

    YTY = np.dot(Y_e.T, Y_e) # precompute this

    R = np.eye(f + 1) # regularization matrix
    R[f, f] = 0 # don't regularize the biases!
    R *= lambda_reg

    YTYpR = YTY + R

    byY = np.dot(b_y, Y_e) # precompute this as well

    X_new = np.zeros((m, f + 1), dtype=dtype)

    for k, s_u, i_u in iter_rows(S):
        Y_u = Y_e[i_u] # exploit sparsity
        b_y_u = b_y[i_u]
        A = np.dot((1 - b_y_u) * s_u + 1, Y_u)
        A -= byY

        YTSY = np.dot(Y_u.T, (Y_u * s_u[:, None]))
        B = YTSY + YTYpR

        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv)
        X_new[k] = np.linalg.solve(B.T, A.T).T # doesn't seem to make much of a difference in terms of speed, but w/e

    return X_new

def factorize(S, num_factors, lambda_reg=1e-5, num_iterations=20, init_std=0.01, verbose=False, dtype='float32', recompute_factors=recompute_factors, *args, **kwargs):
    """
    factorize a given sparse matrix using the Weighted Matrix Factorization algorithm by
    Hu, Koren and Volinsky.
    S: 'surplus' confidence matrix, i.e. C - I where C is the matrix with confidence weights.
        S is sparse while C is not (and the sparsity pattern of S is the same as that of
        the preference matrix, so it doesn't need to be specified separately).
    num_factors: the number of factors.
    lambda_reg: the value of the regularization constant.
    num_iterations: the number of iterations to run the algorithm for. Each iteration consists
        of two steps, one to recompute U given V, and one to recompute V given U.
    init_std: the standard deviation of the Gaussian with which V is initialized.
    verbose: print a bunch of stuff during training, including timing information.
    dtype: the dtype of the resulting factor matrices. Using single precision is recommended,
        it speeds things up a bit.
    recompute_factors: helper function that implements the inner loop.
    returns:
        U, V: factor matrices. If bias=True, the last columns of the matrices contain the biases.
    """
    num_users, num_items = S.shape

    ST = S.T.tocsr()

    U = None # no need to initialize U, it will be overwritten anyway
    V = np.random.randn(num_items, num_factors).astype(dtype) * init_std

    for i in range(num_iterations):
        U = recompute_factors(V, S, lambda_reg, dtype, *args, **kwargs)
        V = recompute_factors(U, ST, lambda_reg, dtype, *args, **kwargs)
    return U, V


# Code from https://github.com/MrChrisJohnson/implicit-mf/blob/master/mf.py
class ImplicitMF():

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in range(self.num_iterations):
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in range(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu

        return solve_vecs


def time_reps(func, params, reps):
    start = time.time()
    for s in range(reps):
        func(*params)
    print("Executed " + str(func))
    return (time.time() - start) / reps


"""
Compare and contrast different methods of computing the SVD
"""

k = 5
reps = 1


# Define SVDs
def scipy_svd(A, K):
    return svds(A, k, return_singular_vectors=False)

def sklearn_randomized_svd(A, k):
    return randomized_svd(A, k)

def sklearn_truncated_randomized_svd(A, k):
    return TruncatedSVD(A, k, n_components=k, algorithm='randomized')

def sklearn_truncated_arpack_svd(A, k):
    return TruncatedSVD(A, k, n_components=k, algorithm='arpack')

def sparsesvd_svd(A, k):
    return sparsesvd(A, k)

def gensim_svd(A, k):
    return stochastic_svd(A, k, A.shape[0])

# Define matrix fact, deep learning and others
def wmf(A, k):
    U, V = factorize(A, num_factors=k, lambda_reg=1e-5, num_iterations=2,
                         recompute_factors=recompute_factors_bias)

def implicit_mf(A, k):
    i = ImplicitMF(A, num_factors=k)

def nmf_1(A, k):
    return NMF_ANLS_BLOCKPIVOT().run(A, k, max_iter=50)

def nmf_2(A, k):
    return NMF_ANLS_AS_NUMPY().run(A, k, max_iter=50)

def nmf_3(A, k):
    return NMF_ANLS_AS_GROUP().run(A, k, max_iter=50)

def nmf_4(A, k):
    return NMF_HALS().run(A, k, max_iter=500)

def nmf_5(A, k):
    return NMF_MU().run(A, k, max_iter=500)

# Daal4py
def daal4py_svd(A, k):  # only dense inputs
    algo = d4p.svd()
    result2 = algo.compute(A)

def daal4py_als(A, k):
    algo1 = d4p.implicit_als_training_init(nFactors=k)
    result1 = algo1.compute(A)
    algo2 = d4p.implicit_als_training(nFactors=k)
    result2 = algo2.compute(A, result1.model)

# Vary the matrix size
def time_ns():
    density = 10**-6 # divide by two as we are symmetrizing the matrix
    ns = np.arange(20, 21) * 10**3  # Change if you want to test a range of matrix sizes
    times = dict()

    for i, n in enumerate(ns):
        # Generate random sparse matrix
        nz = int(n * n * density / 2)  # half the number of non-zero becuase of symmetrization
        inds = np.random.randint(int(n), size=(2, nz))
        data = np.random.rand(nz)
        A = scipy.sparse.csc_matrix((data, inds), (n, n))
        # Symmetrize
        A += A.T

        print('\nMatrix {} x {}'.format(n, n))
        print('Non-zero {}'.format(A.nnz))

        # Run methods
        times['scipy_svd'] = time_reps(scipy_svd, (A, k), reps)  # too slow
        print(times['scipy_svd'])
        times['sklearn_truncated_randomized_svd'] = time_reps(sklearn_truncated_randomized_svd, (A, k), reps)
        print(times['sklearn_truncated_randomized_svd'])
        times['sklearn_truncated_arpack_svd'] = time_reps(sklearn_truncated_arpack_svd, (A, k), reps)
        print(times['sklearn_truncated_randomized_svd'])
        times['sklearn_randomized_svd'] =time_reps(sklearn_randomized_svd, (A, k), reps)
        print(times['sklearn_randomized_svd'] )
#         times['sparsesvd_svd'] = time_reps(sparsesvd_svd, (A, k), reps)  # too slow
#         print(times['sparsesvd_svd'])
        times['gensim_svd'] = time_reps(gensim_svd, (A, k), reps)
        print(times['gensim_svd'])

#         times['wmf'] = time_reps(wmf, (A, k), reps)  # too slow
#         print(times['wmf'])
        times['implicit_mf'] = time_reps(implicit_mf, (A, k), reps)
        print(times['implicit_mf'])
        times['nmf_1'] = time_reps(nmf_1, (A, k), reps)
        print( times['nmf_1'])
#         times['nmf_2']  = time_reps(nmf_2, (A, k), reps)  # too slow
#         print(times['nmf_2'])
        times['nmf_3']  = time_reps(nmf_3, (A, k), reps)
        print(times['nmf_3'])
#         times['nmf_4']  = time_reps(nmf_4, (A, k), reps)  # too slow
#         print(times['nmf_4'])
#         times['nmf_5'] = time_reps(nmf_5, (A, k), reps)  # too slow
#         print(times['nmf_5'])

        # These do not support sparse inputs
#         times['daal4py_asl'] = time_reps(daal4py_asl, (A, k), reps)
#         times['daal4py_svd'] = time_reps(daal4py_svd, (A, k), reps)

        print("Times", times)


if __name__ == '__main__':
    time_ns()

# Output:
# Times {
# 'sklearn_truncated_randomized_svd': 0.0001538991928100586,
# 'sklearn_truncated_arpack_svd': 1.5735626220703125e-05,
# 'sklearn_randomized_svd': 633.9519461393356,
# 'gensim_svd': 62.28390669822693,
# 'implicit_mf': 1.6450881958007812e-05,
# 'nmf_1': 2538.050609588623,
# 'nmf_3': 4463.238362908363}
# 'implicit_mf': 1.6450881958007812e-05,
# 'nmf_1': 2538.050609588623,
# 'nmf_3': 4463.238362908363}
