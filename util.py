import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


def get_A_b(P):
    S = P.shape[0]
    Actions = P.shape[2]
    b = np.zeros(S+1)
    b[-1] = 1
    x = np.full((1, Actions*S), 1)
    A = np.copy(P)
    for i in range(S):
        A[i,i,:]-=1
    A = A.reshape([S,S*Actions])
    A = np.vstack([A, x])
    return A,b


def update_rho(rho,Y,P,delta):
    object_rho = lambda x : -sum(x*Y)+sum(x**2)/2
    object_der = lambda x : x-Y
    d = len(Y)
    lb = [delta]*d
    ub = [1]*d
    A,b = get_A_b(P)
    bounds = Bounds(lb,ub)
    linear_constraint = LinearConstraint(A,b,b)
    x0 = rho
    res = minimize(object_rho, x0, method='trust-constr', jac=object_der, hess=None,
               constraints=[linear_constraint],
               options={'verbose': 0,'maxiter': 100}, bounds=bounds)
    #print(res.x)
    return np.maximum(res.x,delta)


def update_rho_SLSQP(rho,Y,P,delta):
    object_rho = lambda x : -sum(x*Y)+sum(x**2)/2
    object_der = lambda x: x - Y
    d = len(Y)
    lb = [delta]*d
    ub = [1]*d
    A,b = get_A_b(P)
    bounds = Bounds(lb,ub)
    eq_cons = {'type': 'eq',
               'fun': lambda x: np.dot(A,x)-b,
               'jac': lambda x: A}
    x0 = rho
    res = minimize(object_rho, x0, method='SLSQP', jac=object_der, hess=None,
               constraints=[eq_cons],
               options={'verbose': 1,'disp' : 0}, bounds=bounds,)
    #print(res.x)
    return np.maximum(res.x,delta)
