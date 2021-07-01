import operator as op
from functools import reduce

# n choose r
def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

# return the number of points of each weight in a ball of radius r in F_2^n
def weights_ball(n, r):
    w=[comb(n, int(0))]
    for i in range(1,r+1):
        w.append( (w[i-1]*int(n-i+1)) // (int(i)) )
    return w

# weight distribution of e1+e2 with e1, e2 from two balls
def weights_ball_addition(n, r1, r2):
    C = [0 for i in range(min(n+1,r1+r2+1))]
    ball = weights_ball(n,r1)
    for x in range(r1+1):
        for y in range(r2+1):
            for o in range(max(x+y-n, 0),min(x,y)+1):
                C[x+y-2*o] += ball[x]*comb(x,o)*comb(n-x,y-o)
    return C

# convolution of two distributions/measure
def convol(A, B):
    C = [0 for i in range(len(A)+len(B)-1)]  
    for x in range(len(A)):
        for y in range(len(B)):
            C[x+y] += A[x]*B[y]
    return C

# convolution of two distributions bounded by weight w
def bounded_convol(A, B, w):
    C = [0 for i in range(min(w,len(A)+len(B)-1))]  
    for x in range(len(A)):
        for y in range(min(w-x,len(B))):
            C[x+y] += A[x]*B[y]
    return C

def weights_birthday(A,B,ell):
    return convol(A,B), 2**ell

def weights_ISD_generic(n, r, p, ell, weights_subISD):
    k = n-r
    E1 = weights_ball(r-ell, r-ell)
    E2, normalizer = weights_subISD(k+ell, ell, p)
    return convol(E1, E2), 2**(r-ell)*normalizer

subISD_prange = lambda nn,rr,pp: ([1], 1)
subISD_LB = lambda nn,rr,pp: (weights_ball(nn, pp), 2**rr)
subISD_SD = lambda nn,rr,pp: weights_birthday(weights_ball(nn//2, pp//2),weights_ball((nn+1)//2, (pp+1)//2), rr)
def subISD_MMT(ell2):
    def subISD(nn,rr,pp):
        # count (e1+e3||e2+e4) with:
        # e_i from ball(nn//2, pp//4)
        # ell2-dimensional constraint on H(e1+e2)
        # ell2-dimensional constraint on H(e3+e4)
        # (rr-ell2)-dimensional constraint on H(e1+e3||e2+e4)
        assert ell2 <= rr
        p = [pp//4, pp//4+(pp%4>0), pp//4+(pp%4>2), pp//4+(pp%4>1)]
        E13 = weights_ball_addition((nn+1)//2, p[0], p[2])
        E24 = weights_ball_addition(nn//2, p[1], p[3])
        return convol(E13, E24), 2**(rr+ell2)
    return subISD

def weights_prange(n,r,ell=0):
    return weights_ISD_generic(n,r,0,ell,subISD_prange)

def weights_LB(n, r, p, ell=0):
    return weights_ISD_generic(n,r,p,ell,subISD_LB)

def weights_SD(n, r, p, ell):
    return weights_ISD_generic(n,r,p,ell,subISD_SD)

def weights_MMT(n, r, p, ell, ell2):
    return weights_ISD_generic(n,r,p,ell,subISD_MMT(ell2))

def unique_proba_prange(n, r, w, ell=0):
    W, norm = weights_prange(n, r, ell)
    return W[w]/norm

def random_proba_prange(n, r, w, ell=0):
    W, norm = weights_prange(n, r, ell)
    return sum(W[:(w+1)])/norm

def unique_proba_LB(n, r, w, p=3, ell=0):
    W, norm = weights_LB(n, r, p, ell)
    return W[w]/norm

def random_proba_LB(n, r, w, p=3, ell=0):
    W, norm = weights_LB(n, r, p, ell)
    return sum(W[:(w+1)])/norm