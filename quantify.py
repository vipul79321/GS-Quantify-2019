import math
import random
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

np.random.seed(420) 
def euro(P,K,k):
    # dP = [P[i+1]-P[i-1] for i in range(1,len(P)-1)]
    dP = np.gradient(P)
    # dK = [K[i+1]-K[i-1] for i in range(1,len(K)-1)]
    dK = np.gradient(K)

    F = [0.0]
    for i in range(1,len(dP)-1):
        F.append(dP[i]/dK[i])

    F.append(1.0)
    f = interp1d(F,K)
    f_ = interp1d(K,F)
    N = 1000000
    sts = [max(f(np.random.uniform()) - k, 0) for i in range(N)]
    st = sum(sts)/N
    
    return st

def asia(P1,K1,P2,K2,k):
    # dP1 = [P1[i+1]-P1[i-1] for i in range(1,len(P1)-1)]
    dP1 = np.gradient(P1)
    # dK1 = [K1[i+1]-K1[i-1] for i in range(1,len(K1)-1)]
    dK1 = np.gradient(K1)

    F1 = [0.0]
    for i in range(1,len(dP1)-1):
        F1.append(dP1[i]/dK1[i])

    F1.append(1.0)
    f1 = interp1d(F1,K1)

    # dP2 = [P2[i+1]-P2[i-1] for i in range(1,len(P2)-1)]
    dP2 = np.gradient(P2)
    # dK2 = [K2[i+1]-K2[i-1] for i in range(1,len(K2)-1)]
    dK2 = np.gradient(K2)

    F2 = [0.0]
    for i in range(1,len(dP2)-1):
        F2.append(dP2[i]/dK2[i])

    F2.append(1.0)
    f2 = interp1d(F2,K2)

    N = 100000
    st = 0.0
    for i in range(N):
        u = np.random.uniform()
        a = 1 - u
        t = np.random.uniform()
        b = (-0.95)*(2*a*t + 1) + 2*(0.95*a)**2*t+1
        c = (0.95)**2*(4*a**2*t - 4*a*t + 1) + (0.95)*(4*a*t - 4*a +2)+1
        v = (2*t*(a*0.95 -1)**2)/(b + c**0.5)
        # v = (t*(1-t))/(1 - (0.95)*(1-t)*(1-u))**2
        sts.append((f1(u)+f2(v))/2)
        st += max((f1(u)+f2(v))/2 - k, 0)
    plt.plot(K, sts)
    plt.xlabel("Strike Price")
    plt.ylabel("Stock Price CDF")
    plt.show()
    return st/N

type_opt = input()
if type_opt == "E":
    k,N = map(int, input().split()) 
    K = []
    P = []
    for i in range(N+1):
        inp = input().split() 
        K.append(float(inp[0]))
        P.append(float(inp[1]))
    


    print(euro(P,K,k))

if type_opt == "A":
    k,M,N = map(int, input().split()) 
    K1 = []
    P1 = []
    for i in range(M+1):
        inp = input().split() 
        K1.append(float(inp[0]))
        P1.append(float(inp[1]))
    
    K2 = []
    P2 = []
    for i in range(N+1):
        inp = input().split() 
        K2.append(float(inp[0]))
        P2.append(float(inp[1]))
    

    print(asia(P1,K1,P2,K2,k))
