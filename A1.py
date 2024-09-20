#Question 1
import numpy as np
# from numpy.linalg
from cvxopt import matrix, solvers

x = np.array([[1,2,3],[4,5,6]])
print(-x)

solvers.options['show progress'] = False

def minimizeL2(X, y):
    #when solving, the inverse A^-1 is left multiplied, therefore included in the function
    return np.linalg.solve(X.T @ X, X.T @ y)

def minimizeL1(X, y):
    n, d = X.shape
    zeroVd = np.zeros(d)
    onesV = np.ones(n)
    c = np.concatenate((onesV, zeroVd), axis=0)

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

    print(y.shape)
    print(h1.shape)

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
    print(res)
    return res ["x"][:d]

def loss_L2(w, X, y):
    return np.mean(((X@w - y)**2)/2)
    
def loss_L1(w, X, y):
    return np.mean(abs(X@w - y))

def loss_Linf(w, X, y):
    return max(abs(X@w - y))




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
        # Xtrain, ytrain = genData(n_train)
        # Xtest, ytest = genData(n_test)
        data = np.loadtxt('toy_data/regression_train.csv', delimiter=',', skiprows=1)

        Xtrain = np.array( data[:, 0]) 
        ytrain = np.array( data[:, 1])

        print(Xtrain, ytrain)

        data = np.loadtxt('toy_data/regression_test.csv', delimiter=',', skiprows=1)
        Xtest=np.array( data[:, 0])
        ytest=np.array( data[:, 1])

        # Learn different models from the training data
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        train_loss1_w1 = loss_L1(w = w_L1, x=Xtrain, y=ytrain)
        train_loss2_w1 = loss_L2(w = w_L1, x=Xtrain, y=ytrain)
        train_lossL_w1 = loss_Linf(w = w_L1, x=Xtrain, y=ytrain)

        train_loss1_w2 = loss_L1(w = w_L2, x=Xtrain, y=ytrain)
        train_loss2_w2 = loss_L2(w = w_L2, x=Xtrain, y=ytrain)
        train_lossL_w2 = loss_Linf(w = w_L2, x=Xtrain, y=ytrain)

        train_loss1_w3 = loss_L1(w = w_Linf, x=Xtrain, y=ytrain)
        train_loss2_w3 = loss_L2(w = w_Linf, x=Xtrain, y=ytrain)
        train_lossL_w3 = loss_Linf(w = w_Linf, x=Xtrain, y=ytrain)

        train_loss2 = np.concatenate(train_loss2_w2 , train_loss2_w1, train_loss2_w3)
        train_loss1 = np.concatenate(train_loss1_w2, train_loss1_w1, train_loss1_w3)
        train_lossinf = np.concatenate(train_lossL_w2,train_lossL_w1,  train_lossL_w3)
        
        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the training
        # data). Save them to `train_loss`
        train_loss[r] = np.concatenate((train_loss2 , train_loss1 , train_lossinf), axis = 1)

        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the test
        # data). Save them to `test_loss`
        
        test_loss1_w1 = loss_L1(w = w_L1, x=Xtest, y=ytest)
        test_loss2_w1 = loss_L2(w = w_L1, x=Xtest, y=ytest)
        test_lossL_w1 = loss_Linf(w = w_L1, x=Xtest, y=ytest)

        test_loss1_w2 = loss_L1(w = w_L2, x=Xtest, y=ytest)
        test_loss2_w2 = loss_L2(w = w_L2, x=Xtest, y=ytest)
        test_lossL_w2 = loss_Linf(w = w_L2, x=Xtest, y=ytest)

        test_loss1_w3 = loss_L1(w = w_Linf, x=Xtest, y=ytest)
        test_loss2_w3 = loss_L2(w = w_Linf, x=Xtest, y=ytest)
        test_lossL_w3 = loss_Linf(w = w_Linf, x=Xtest, y=ytest)

        test_loss2 = np.concatenate(test_loss2_w2 , test_loss2_w1, test_loss2_w3)
        test_loss1 = np.concatenate(test_loss1_w2, test_loss1_w1, test_loss1_w3)
        test_lossinf = np.concatenate(test_lossL_w2,test_lossL_w1,  test_lossL_w3)

        test_loss[r] = np.concatenate((test_loss2 , test_loss1 , test_lossinf), axis = 1)
        
    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)



print(synRegExperiments())

#E)
# All Linf losses are much larger, as they are skewed towards outliers
# L2 loss is larger than L1 because more sensitive to outliers than L1, since it grows exponentially vs abs value
# Linf loss works the best with Linf model


# def compute_reg_losses(w_L2, w_L1, w_Linf, X, y):



# def preprocessAutoMPG(dataset_folder):

# def runAutoMPG(dataset_folder):

# def linearRegL2Obj(w, X, y):

# def gd(func, w_init, X, y, step_size, max_iter, tol=1e-10):    