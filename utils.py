import numpy as np
from numpy.linalg import inv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import statsmodels.api as sm


def sigmoid(x):
    return 1/(1+np.exp(-x))


def all_same(list1, list2):
    for i in range(len(list1)):
        if abs(list1[i] - list2[i]) > 1e-5:
            return False 
    return True 


def linear_loo(X, eps, i, H, beta):
    return beta - (inv(X.T @ X) @ X[i,:].reshape((-1, 1)) @ eps[i]).reshape((-1, 1)) / (1-H[i,i])


def ridge_loo(X, eps, i, H, beta, alpha):
    return beta - (inv(X.T @ X + alpha * np.identity(X.shape[1])) @ X[i,:].reshape((-1, 1)) @ eps[i]).reshape((-1, 1)) / (1-H[i,i])


def linear_press(X, y):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    I = np.identity(n)
    H = X @ inv(X.T @ X) @ X.T
    rss = y.T @ (I - H) @ np.diag(np.diag(I - H) ** (-2)) @ (I - H) @ y 
    return rss.item()


def linear_loocv(X, y):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    eps = []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        beta_i = inv(X_i.T @ X_i) @ X_i.T @ y_i     
        eps.append( y[i] - X[i] @ beta_i ) 
    return np.sum(np.array(eps) ** 2)


def ridge_press(X, y, alpha):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    I = np.identity(n)
    H = X @ inv(X.T @ X + alpha * np.identity(p)) @ X.T 
    rss = y.T @ (I - H) @ np.diag(np.diag(I - H) ** (-2)) @ (I - H) @ y 
    return rss.item()


def ridge_loocv(X, y, alpha):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    eps = []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        beta_i = inv(X_i.T @ X_i + alpha * np.identity(p)) @ X_i.T @ y_i     
        eps.append( y[i] - X[i] @ beta_i ) 
    return np.sum(np.array(eps) ** 2)


def logistic_approx(X, y):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    beta = LogisticRegression().fit(X, y.ravel()).coef_.reshape((-1, 1))
    ps = np.sqrt(sigmoid(X @ beta))
    W_sqrt = np.diag(ps.flatten())
    Z = W_sqrt @ X 
    H = Z @ (Z.T @ Z) @ Z.T
    eps = W_sqrt @ (y - ps.reshape((-1, 1)))

    beta_approx, err = [], []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        beta_i_loo = linear_loo(Z, eps, i, H, beta.reshape((-1, 1)))
        p_i = sigmoid(X_i @ beta_i_loo)
        err.append(log_loss(y_i, p_i))
        beta_approx.append(beta_i_loo)
    return beta_approx, err
    

def logistic_loocv(X, y):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n = X.shape[0]
    beta_loocv, err = [], []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        beta_i = LogisticRegression(max_iter=1000).fit(X_i, y_i.ravel()).coef_.reshape((-1, 1))
        p_i = sigmoid(X_i @ beta_i)
        err.append(log_loss(y_i, p_i))
        beta_loocv.append(beta_i)
    return beta_loocv, err


def logistic_ridge_approx(X, y, alpha):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    beta = LogisticRegression(penalty="l2", C=1/alpha, fit_intercept=False, max_iter=1000).fit(X, y.ravel()).coef_.reshape((-1, 1))
    ps = np.sqrt(sigmoid(X @ beta))
    W_sqrt = np.diag(ps.flatten())
    Z = W_sqrt @ X 
    H = Z @ (Z.T @ Z + 2 * alpha * np.identity(p)) @ Z.T
    eps = W_sqrt @ (y - ps.reshape((-1, 1)))

    beta_approx, err = [], []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        beta_i_loo = ridge_loo(Z, eps, i, H, beta.reshape((-1, 1)), 2*alpha)
        p_i = sigmoid(X_i @ beta_i_loo)
        err.append(log_loss(y_i, p_i))
        beta_approx.append(beta_i_loo)
    return beta_approx, err


def logistic_ridge_loocv(X, y, alpha):
    X, y = np.array(X), np.array(y).reshape((-1, 1))
    n, p = X.shape
    logistic_model = LogisticRegression(penalty="l2", C=1/alpha, fit_intercept=False)

    beta_loocv, err = [], []
    for i in range(n):
        X_i, y_i = X[np.arange(X.shape[0]) != i], y[np.arange(y.shape[0]) != i]
        logistic_model.fit(X_i, y_i.ravel())
        beta_i = logistic_model.coef_.reshape((-1, 1))
        p_i = sigmoid(X_i @ beta_i)
        err.append(log_loss(y_i, p_i))
        beta_loocv.append(beta_i)
    return beta_loocv, err
    