# Functions to generate or download the challenges of decodingchallenge.org

import random
import math
import numpy as np
import os
import requests
import logging

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

def random_parity(n, k):
    P=np.zeros((k, n-k), dtype='bool')
    for i in range(k):
        for j in range(n-k):
            P[i,j] = random.randint(0,1)
    return P

# extraced from https://decodingchallenge.org/Scripts/lowweight_generate.py
def challenge_LW_G(n, seed):
    k = n//2
    random.seed(seed)
    G=np.zeros((k, n), dtype='bool')
    G[:, k:] = random_parity(n, k)
    for i in range(k):
        G[i,i]=1
    return G

def challenge_LW_H(n, seed):
    k = n//2
    random.seed(seed)
    H=np.zeros((n-k, n), dtype='bool')
    H[:n-k, :k] = np.transpose(random_parity(n, k))
    for i in range(n-k):
        H[i,k+i]=1
    return H

# extracted from https://decodingchallenge.org/Scripts/syndrome_generate.py
def challenge_SD_G(n, seed):
    k = n//2
    G=challenge_LW_G(n, seed)
    t=np.zeros((n,), dtype='bool')
    for i in range(n-k):
        t[k+i] = random.randint(0,1)
    return G, t, w

def challenge_SD_H(n, seed):
    H=challenge_LW_H(n, seed)
    k = n//2
    w = math.ceil(1.05 * dGV(n,k))
    s=np.zeros((n-k,), dtype='bool')
    for i in range(n-k):
        s[i] = random.randint(0,1)
    return H, s, w

# Large Weight Syndrome Decoding Problem
goppa_instances = [20,48,80,117,156,197,240,286,333,381,431,482,534,587,640,695,751,808,865,923,982,1041,1101,1161,1223,1284,1347,1409,1473,1536,1600,1665,1730,1796,1862,1928,1995,2062,2129,2197,2265,2334,2403,2472,2541,2611,2681,2752,2822,2893,2965,3036,3108,3180,3253,3325,3398,3471,3545,3618,3692,3766,3840,3915,3990,4065,4140,4215,4291,4367,4443,4519,4595,4672,4749,4826,4903,4980,5058,5136,5214,5292,5370,5448,5527,5606,5685,5764,5843,5923,6002,6082,6162,6242,6322,6402]
def download_goppa(n):
    if n not in goppa_instances:
        nextn = next(x[1] for x in enumerate(goppa_instances) if x[1] > n)
        print('No Goppa instance with such parameter exists, returning next larger: {}'.format(nextn))
        return False

    filename = "challenges/Goppa-{}.txt".format(n)
    if not os.path.isdir("challenges"):
        os.mkdir("challenges")

    if os.path.isfile(filename) is False:
        logging.info("Did not find '{filename}', downloading ...".format(filename=filename))
        url = "https://decodingchallenge.org/Challenges/Goppa/Provider0/Goppa_{}".format(n)
        r = requests.get(url)
        logging.info("%s %s" % (r.status_code, r.reason))
        fn = open(filename, "w")
        fn.write(r.text)
        fn.close()
    return filename


def load_goppa_from_file(filename):
    with open(filename) as fp:
        fp.readline() # n
        n = int(fp.readline())
        fp.readline() # k
        k = int(fp.readline())
        fp.readline() # w
        w = int(fp.readline())
        fp.readline() # H^transpose ...
        P=np.zeros((k, n-k), dtype='bool')
        for i in range(k):
            line=fp.readline()
            P[i] = np.array(list(map(int,line[:n-k])), dtype='bool')
        fp.readline() # s^transpose
        line=fp.readline()
        s = np.array(list(map(int,line[:n-k])), dtype='bool')
    return P, s, k, w
    
def challenge_goppa_G(n):
    filename=download_goppa(n)
    print(filename)
    if filename:
        P, s, k, w = load_goppa_from_file(filename)
        G=np.zeros((k, n), dtype='bool')
        G[:, k:] = P
        for i in range(k):
            G[i,i]=1
        t=np.zeros((n,), dtype='bool')
        for i in range(n-k):
            t[k+i] = s[i]
        return G, t, k, w
    return False

def challenge_goppa_H(n):
    filename=download_goppa(n)
    print(filename)
    if filename:
        P, s, k, w = load_goppa_from_file(filename)
        H=np.zeros((n-k, n), dtype='bool')
        H[:n-k, :k] = np.transpose(P)
        for i in range(n-k):
            H[i,k+i]=1
        return H, s, k, w
    return False



