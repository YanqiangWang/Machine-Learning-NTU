import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    
    for i in range(0, m):
        cost += (y[i] - theta[1] * X[i][1] - theta[0]) ** 2
    cost /= 2 * m

    # ==========================================================

    return cost
