#!/usr/bin/env python
# coding: utf-8

import operator as op
from functools import reduce
from math import ceil
from probability import unique_proba_prange, random_proba_prange, unique_proba_LB, random_proba_LB


# ## Tools

# n choose r
def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def dGV(n, k):
    d = 0
    aux = 2**(n-k)
    b = 1
    while aux >= 0:
        aux -= b
        d += 1
        b *= (n-d+1)
        b /= d
    return d 

def param_SD(n):
    k = n//2
    w = int(ceil(1.05 * dGV(n,k)))
    return (k,w)


# ## Probabilities

# probability that a binary n x n matrix is invertible
def proba_invertible(n):
    return reduce(op.mul, [1-2**(-k) for k in range(1,n+1)], 1)

# probability of successfully finding the error (provided the matrix is invertible)
# def proba_succ_prange(n,k,w):
#     return comb(n-k,w) / min(comb(n,w),2**(n-k))

# def proba_tot_prange(n,k,w):
#     return proba_invertible(n-k) * proba_succ_prange(n,k,w)

# def proba_succ_LB(n,k,w,p):
#     return min(1,reduce(op.add, [comb(n-k,w-i)*comb(k,i) for i in range(0,p+1)])/ min(comb(n,w),2**(n-k)))

# def proba_tot_LB(n,k,w,p):
#     return proba_invertible(n-k) * proba_succ_LB(n,k,w,p)


# ## Tests

def get_all_proba(n):
    k,w = param_SD(n)
    p = 3
    return (proba_succ_prange(n,k,w), proba_tot_prange(n,k,w), proba_succ_LB(n,k,w,p), proba_tot_LB(n,k,w,p))

def print_all_proba(n, w_unique=False):
    k,w = param_SD(n)
    p = 3
    print("n =", n)
    print("k =", k)
    print("w =", w)
    print("p =", p)
    print("----Random Decoding----")
    print("random_proba_prange = \t",random_proba_prange(n,k,w))
    #print("Proba_tot_Prange = \t",proba_invertible(n-k)*random_proba_prange(n,k,w))
    print("random_proba_LB = \t",random_proba_LB(n,k,w,p))
    #print("Proba_tot_LB = \t\t",proba_invertible(n-k)*random_proba_LB(n,k,w,p))
    if(w_unique):
        print("----Unique Decoding at weight "+str(w_unique)+" ----")
        print("unique_proba_prange = \t",unique_proba_prange(n,k,w_unique))
        print("unique_proba_LB = \t",unique_proba_LB(n,k,w_unique,p))





