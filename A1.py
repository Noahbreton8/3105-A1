#Question 1
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import os

# x = np.array([[1,2,3],[4,5,6]])
# print(-x)

solvers.options['show_progress'] = False

def minimizeL2(X, y):
    #when solving, the inverse A^-1 is left multiplied, therefore included in the function
    return np.linalg.solve(X.T @ X, X.T @ y)

def minimizeL1(X, y):
    n, d = X.shape
    zeroVd = np.zeros(d)
    onesV = np.ones(n)
    c = np.concatenate((zeroVd, onesV), axis=0)

    zerosM = np.zeros(X.shape)
    negEye = -np.eye(N=n)
    g1 = np.concatenate((zerosM, negEye), axis=1)
    g2 = np.concatenate((X, negEye), axis =1)
    g3 = np.concatenate((-X, negEye), axis=1)

    G = np.concatenate((g1,g2,g3),axis = 0)

    zeroVn = np.zeros_like(y)
    h1 = zeroVn
    h2 = y
    h3 = -y

    # print(y.shape)
    # print(h1.shape)

    H = np.concatenate((h1,h2,h3), axis = 0)

    c = matrix(c)
    G = matrix(G)
    H = matrix(H)


    res = solvers.lp(c,G,H)
    return res["x"][:d]

def minimizeLinf(X, y):
    n, d = X.shape
    zeroVd = np.zeros(d)
    onesV = np.ones((n,1))

    c = np.concatenate([zeroVd, [1]], axis = 0)

    g1 = np.concatenate((zeroVd, [-1]))
    g1 = np.array([g1])

    # print(g1)

    g21 = X
    g22 = -onesV
    g2 = np.concatenate([g21, g22], axis = 1)

    g31 = -X
    g32 = -onesV
    g3 = np.concatenate([g31, g32], axis = 1)

    # print(g1.shape)
    # print(g2.shape)
    # print(g3.shape)

    G = np.concatenate([g1,g2,g3], axis = 0)

    h1 = [[0]]
    h2 = y
    h3 = -y

    H = np.concatenate([h1, h2, h3], axis =0)

    c = matrix(c)
    G = matrix(G)
    H = matrix(H)

    res = solvers.lp(c,G,H)
    # print(res)
    return res ["x"][:d]

def loss_L2(w, X, y):
    return np.mean(((X@w - y)**2)/2)
    
def loss_L1(w, X, y):
    return np.mean(abs(X@w - y))

def loss_Linf(w, X, y):
    return max(abs(X@w - y))

def compute_reg_losses(w_L2, w_L1, w_Linf, X, y):
    np_results = np.zeros([3, 3])

    np_results[0, 0] = loss_L2(w_L2, X, y)
    np_results[1, 0] = loss_L2(w_L1, X, y)
    np_results[2, 0] = loss_L2(w_Linf, X, y)

    np_results[0, 1] = loss_L1(w_L2, X, y)
    np_results[1, 1] = loss_L1(w_L1, X, y)
    np_results[2, 1] = loss_L1(w_Linf, X, y)

    np_results[0, 2] = loss_Linf(w_L2, X, y)
    np_results[1, 2] = loss_Linf(w_L1, X, y)
    np_results[2, 2] = loss_Linf(w_Linf, X, y)

    return np_results


def synRegExperiments():
    def genData(n_points):
        '''
        This function generate synthetic data
        '''
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        return X, y
    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train)
        Xtest, ytest = genData(n_test)
        # data = np.loadtxt('toy_data/regression_train.csv', delimiter=',', skiprows=1)

        # Xtrain, ytrain= data.T

        # Xtrain = Xtrain.reshape(-1, 1)
        # ytrain = ytrain.reshape(-1, 1)
        # Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1) # augment input

        # # print(Xtrain.shape, ytrain.shape)
        # dataT = np.loadtxt('toy_data/regression_test.csv', delimiter=',', skiprows=1)
        # Xtest, ytest = dataT.T
        
        # Xtest = Xtest.reshape(-1, 1)
        # Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1) # augment input

        # ytest = ytest.reshape(-1, 1)

        # Learn different models from the training data
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # print(w_L2, w_L1, w_Linf)

        train_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtrain, ytrain)

        test_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtest, ytest)

    return np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)

# print(synRegExperiments())

#1e)
# All Linf losses are much larger, as they are skewed towards outliers
# L2 loss is larger than L1 because more sensitive to outliers than L1, since it grows exponentially vs abs value
# Linf loss works the best with Linf model, due to model minimizing worst outlier and loss checking the worst outlier, this will match up better than L2 or L1 loss
# Linf loss does'nt work as well with the L1 and L2 Models, because the models dont correct as much for outliers, the line of best fit will not be close to that point


# Q2b
def gd(func, w_init, X, y, step_size, max_iter, tol=1e-10):

    w = w_init

    for _ in range(max_iter):
        obj_val, grad = func(w, X, y)
        if np.linalg.norm(grad) < tol:
            break
        w = w - step_size * grad

    return w


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


# Q2c
def logisticRegObj(w, X, y):
    # X: n x d
    # y: n x 1 (in {0, 1})
    # Note for over/underflow
    zhat = X @ w
    yhat = sigmoid(zhat)

    obj_val = np.mean(y * np.logaddexp(0, -zhat)
                      + (1 - y) * np.logaddexp(0, zhat))
    grad = (X.T @ (yhat - y) ) / X.shape[0]
    return obj_val, grad


# def linearRegL2Obj(w, X, y):
#     obj_val = np.mean(((X@w - y)**2))/2
#     grad = 1/X.shape[0] * X.T *(X@w -y)

#     jfunc = lambda w: 1/X.shape[0] * X.T *(X@w -y)
#     cpgrad = autograd.grad(jfunc(w))

#     print(grad, cpgrad)
#     return obj_val, grad

    

# def l2Norm (grad):
#     return np.linalg.norm(grad)


# def gd(obj_func, w_init, X, y, eta, max_iter, tol):
#     # TODO: initialize w
#     w = w_init
#     for _ in range(max_iter):
#         obj_val, grad = obj_func(w, X, y)
        
#         if l2Norm(grad) < tol:
#             break
#         w = w - eta * grad
#     # TODO: Compute the gradient of the obj_func at current w
#     # TODO: Break if the L2 norm of the gradient is smaller than tol
#     # TODO: Perform gradient descent update to w
#     return w

# def sigmoid (w,X):
#     return 1/(1+np.e**-(X@w))

# def logisticRegObj(w, X, y):

#     t1 = -y.T * np.log(sigmoid(w,X))
#     t2 = (np.ones_like(X.shape[0]) - y).T
#     t3 = np.log(np.ones_like(X.shape[0]) - sigmoid(w,X))

#     obj_val = 1/X.shape[0] *(t1-t2*t3)

#     grad = 1/X.shape[0] * X.T @ (sigmoid(w,X) -y)

#     return obj_val, grad 


def synClsExperiments():
    def genData(n_points, d):
        '''
        This function generate synthetic data
        '''
        c0 = np.ones([1, d]) # class 0 center
        c1 = -np.ones([1, d]) # class 1 center
        X0 = np.random.randn(n_points, d) + c0 # class 0 input
        X1 = np.random.randn(n_points, d) + c1 # class 1 input
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y
    def runClsExp(m=100, d=2, eta=0.1, max_iter=1000, tol=1e-10):
        '''
        Run classification experiment with the specified arguments
        '''
        Xtrain, ytrain = genData(m, d)
        n_test = 1000
        Xtest, ytest = genData(n_test, d)

        # print(Xtrain.shape, ytrain.shape)
        #print(Xtrain)

        w_init = np.zeros([d+1, 1])
        
        # data = np.loadtxt('toy_data/classification_train.csv', delimiter=',', skiprows=1)
        # # Extract columns
        # x1, x2, y = data.T
        # X0 = x1.reshape(-1, 1)  # Reshape to ensure it's 2D
        # X1 = x2.reshape(-1, 1)  # Reshape to ensure it's 2D
        # Xtrain = np.concatenate((X0, X1), axis=1)
        # Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1)), Xtrain), axis=1)
        # ytrain = y
        # ytrain = ytrain.reshape(-1, 1)

        # dataT = np.loadtxt('toy_data/classification_test.csv', delimiter=',', skiprows=1)
        # x1, x2, y = dataT.T
        # X0 = x1.reshape(-1, 1)  # Reshape to ensure it's 2D
        # X1 = x2.reshape(-1, 1)  # Reshape to ensure it's 2D
        # Xtest = np.concatenate((X0, X1), axis=1)
        # Xtest = np.concatenate((np.ones((Xtest.shape[0], 1)), Xtest), axis=1)
        # ytest = y
        # ytest = ytest.reshape(-1, 1)

        w_logit = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter, tol)
        #print(w_logit)

        #         ytrain_hat = 0.5 * (1 + np.sign(Xtrain @ w_logit))
        # train_acc = np.sum(ytrain == ytrain_hat) / (Xtrain.shape[0])
        # ytest_hat = 0.5 * (1 + np.sign(Xtest @ w_logit))
        # test_acc = np.sum(ytest == ytest_hat) / (Xtest.shape[0])
        # return train_acc, test_acc
    

        #w_logit = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter, tol)
        ytrain_hat = 0.5 * (1 + np.sign(Xtrain @ w_logit))
        train_acc = np.sum(ytrain == ytrain_hat) / (2 * m)

        ytest_hat = 0.5 * (1 + np.sign(Xtest @ w_logit))
        test_acc = np.sum(ytest == ytest_hat) / (2 * n_test)
        return train_acc, test_acc
    
    n_runs = 100
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])
    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, d in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(d=d)
        for i, eta in enumerate((0.1, 1.0, 10., 100.)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(eta=eta)
        # TODO: compute the average accuracies over runs
        # TODO: return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable

    return np.mean(train_acc, axis=0), np.mean(test_acc, axis=0)

# print(synClsExperiments())

#2e)
#for training, the more data points (the larger m is) accuracy plateaus around 0.92
#for test, the more data points the more accuracte prediction
#having more data points helps the model generalize better

#for training, the more features (d) allows it to separate the classes into more distinct groups
#for test, the more features (d) also allows it to separate into more distinct groups

#for small training step size (0.1, 1, 10), the model is fairly accurate
#when training rate is too high, it starts to overshoot predictions

def preprocessAutoMPG(dataset_folder):

    df_data = pd.read_csv(os.path.join(dataset_folder, "auto-mpg.data"), 
                          header=None, 
                          delim_whitespace=True)
    # Drop `origin`` and `car name` features
    del df_data[8]
    del df_data[7]
    # Drop points with missing values: 32, 126, 330, 336, 354, 374
    df_data = df_data.drop([32, 126, 330, 336, 354, 374])

    # Convert labels and then remove them from inputs
    y = df_data[0].to_numpy(float)[:, None]
    del df_data[0]
    X = df_data.to_numpy(float)

    return X, y


def runAutoMPG(dataset_folder):
    X, y = preprocessAutoMPG(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 100
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    for r in range(n_runs):
    # TODO: Randomly partition the dataset into two parts (50%
    # training and 50% test)
        n_train = round(n * 0.5)
        indices = np.random.permutation(n)
        training_idx, test_idx = indices[:n_train], indices[n_train:]
        Xtrain, Xtest = X[training_idx,:], X[test_idx,:]
        ytrain, ytest = y[training_idx,:], y[test_idx,:]
    
    # TODO: Learn three different models from the training data
    # using L1, L2 and L infinity losses
        w_l1 = minimizeL1(X,y)
        w_l2 = minimizeL2(X,y)
        w_linf = minimizeLinf(X,y)
    # TODO: Evaluate the three models' performance (for each model,
    # calculate the L2, L1 and L infinity losses on the training
    # data). Save them to `train_loss`
            # print(w_L2, w_L1, w_Linf)

        train_loss[r] = compute_reg_losses(w_l2, w_l1, w_linf, Xtrain, ytrain)
        
    # TODO: Evaluate the three models' performance (for each model,
    # calculate the L2, L1 and L infinity losses on the test
    # data). Save them to `test_loss`
        test_loss[r] = compute_reg_losses(w_l2, w_l1, w_linf, Xtest, ytest)

    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)

# print(runAutoMPG('auto_data'))

# Q2f
def preprocessSonar(dataset_folder):

    df_data = pd.read_csv(os.path.join(dataset_folder, "sonar.all-data"), 
                          header=None)
    # Convert labels
    df_data[60] = (df_data[60] != 'R').astype(float)
    y = df_data[60].to_numpy()[:, None]
    # Then remove labels from the data
    del df_data[60]
    X = df_data.to_numpy()
    print(y)

    return X, y

# preprocessSonar('sonar_data')

# Q2g
def runSonar(dataset_folder):

    X, y = preprocessSonar(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment

    eta_list = [0.1, 1, 10, 100]
    train_acc = np.zeros([len(eta_list)])
    val_acc = np.zeros([len(eta_list)])
    test_acc = np.zeros([len(eta_list)])

    # TODO: Randomly partition the dataset into three parts (40%
    #       training (use the round function), 40% validation and 
    #       the remaining ~20% as test)
    n_train = round(n * 0.4)
    n_val = round(n * 0.4)
    indices = np.random.permutation(n)
    training_idx, val_idx, test_idx = indices[:n_train], indices[n_train:(n_train+n_val)], indices[(n_train+n_val):]
    Xtrain, Xval, Xtest = X[training_idx,:], X[val_idx,:], X[test_idx,:]
    ytrain, yval, ytest = y[training_idx,:], y[val_idx,:], y[test_idx,:]

    for i, eta in enumerate(eta_list):

        w_init = np.zeros([d + 1, 1])
        w = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter=1000, tol=1e-8)

        # TODO: Evaluate the model's accuracy on the training
        #       data. Save it to `train_acc`
        y_hat = 0.5 * (1 + np.sign(Xtrain @ w))
        train_acc[i] = np.sum(ytrain == y_hat) / Xtrain.shape[0]

        # TODO: Evaluate the model's accuracy on the validation
        #       data. Save it to `val_acc`
        y_hat = 0.5 * (1 + np.sign(Xval @ w))
        val_acc[i] = np.sum(yval == y_hat) / Xval.shape[0]

        # TODO: Evaluate the model's accuracy on the test
        #       data. Save it to `test_acc`
        y_hat = 0.5 * (1 + np.sign(Xtest @ w))
        test_acc[i] = np.sum(ytest == y_hat) / Xtest.shape[0]

    return train_acc, val_acc, test_acc

# print(runSonar('sonar_data'))