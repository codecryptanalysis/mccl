#!/usr/bin/env python3.8

import sys
import random

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def usage():
    eprint("ERROR the script expects 3 arguments, 'n', 'seed' and 'error_string'.")
    eprint("\b - 'n' is an integer corresponding to the size of the matrix: H will be of size n//2 * n")
    eprint("\b - 'seed' is an integer corresponding to the initial value of the random seed")
    eprint("\b - 'error_string' is a string of 0s and 1s of length n//2")
    eprint("This script returns 0 if H*e = s and 1 otherwise.")

def str_to_list(string):
    L = []
    for c in string:
        if c=='0':
            L.append(0)
        elif c=='1':
            L.append(1)
        else:
            raise ValueError("ERROR str_to_list: the string contains a character different from 0 or 1.")
    return(L)

def main(n, seed, error):
    filename = "tests/data/SD_"+str(n)+"_0"
    try:
        file = open(filename,"r")
    except:
        eprint("ERROR file not found")
        usage()
        return 1
    
    file.readline()
    nn = int(file.readline())

    assert nn == n

    if len(error)!=n:
        eprint("ERROR error vector not of length n.")
        usage()
        return 1
    
    file.readline()
    file.readline()
    file.readline()

    s_check = [0]*(n//2)

    for i in range(n//2):
        if error[i]==1:
            s_check[i] = (s_check[i]+1) % 2
    for i in range(n-n//2):
        Hline = file.readline()[:-1]
        if error[n//2+i]==1:
            L = str_to_list(Hline)
            for j in range(n//2):
                s_check[j] = (s_check[j] + L[j]) % 2                

    file.readline()
    s = str_to_list(file.readline()[:-1])
    if s==[]:
        return 1

    file.close()
    
    if s_check == s:
        print("TRUE")
        return 0
    print("FALSE")
    return 1

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv)!=3:
        usage()
        exit()
    try:
        n = int(sys.argv[1])
        error = str_to_list(sys.argv[2])
    except:
        usage()
        exit()
    main(n, 0, error)
