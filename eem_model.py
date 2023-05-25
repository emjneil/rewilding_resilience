# ---- Run the Lotka-Volterra model ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import numpy.matlib
from geneticalgorithm import geneticalgorithm as ga
import seaborn as sns


species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']

# calculate the equilibrium
def find_equilibrium(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0

    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0 

    # return
    return X * (r + np.matmul(A, X))



# run Lotka-Volterra
def Lotka_Volterra(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0

    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0
    return X * (r + np.matmul(A, X))