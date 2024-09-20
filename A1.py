#Question 1
import numpy as np
from numpy.linalg

def minimizeL2(X, y):
    #when solving, the inverse A^-1 is left multiplied, therefore included in the function
    return np.linalg.solve(X.T @ X, X.T @ y)

def minimizeL1(X, y):
    w = np.zeros(len(X[0]))

    #concate w and delta

def minimizeLinf(X, y):
    

def loss_L2(w, X, y):

def loss_L1(w, X, y):

def loss_Linf(w, X, y):

def compute_reg_losses(w_L2, w_L1, w_Linf, X, y):

def synRegExperiments():

def preprocessAutoMPG(dataset_folder):

def runAutoMPG(dataset_folder):

def linearRegL2Obj(w, X, y):

def gd(func, w_init, X, y, step_size, max_iter, tol=1e-10):    