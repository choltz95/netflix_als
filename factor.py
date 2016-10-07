import numpy as np
np.seterr(divide='ignore', invalid='ignore') # supress divide by zero warning

def format_output(I,M,U):
    approximation = np.multiply(I,np.dot(U.T,M))
    with open('output.txt','w') as f:  
        for i in range(943): # users
            for j in range(1682): # movies
                if approximation[i,j] != 0:
                    f.write("%d\t%d\t%f\n"%(i+1,j+1,approximation[i,j]))

def rmse(I,R,M,U,round=False):
    return np.sqrt(np.sum(np.square(np.multiply(I, (R-np.dot(U.T,M)))))/np.count_nonzero(I))

num_users = 943
num_movies = 1682

with open("hw2-training.txt") as f:
    tr = np.loadtxt(f,usecols=range(0,3),dtype=np.int)
with open("hw2-testing.txt") as f:
    tst = np.loadtxt(f,usecols=range(0,3),dtype=np.int)

R = np.asmatrix(np.zeros((num_users,num_movies), dtype=np.int)) # sparse rating matrix
for row in tr:
    user = row[0] - 1  # users start with id 1
    movie = row[1] - 1
    rating = row[2]
    R[user,movie] = rating

Rtst = np.asmatrix(np.zeros((num_users,num_movies), dtype=np.int))
for row in tst:
    user = row[0] - 1
    movie = row[1] - 1
    rating = row[2]
    Rtst[user,movie] = rating

# Indicator Matrices
I = np.copy(R)
I[I > 0] = 1
I[I == 0] = 0
I2 = np.copy(Rtst)
I2[I2 > 0] = 1
I2[I2 == 0] = 0

# ALS Parameters
l = 0.1       # Regularization parameter lambda
k = 0         # convergence iterator
K = 20        # convergence criterion
r = 20        # Dimensionality of latent feature space
m,n = R.shape # Number of users and items

# Initialize matricies
U = 1 * np.random.rand(r,m) # Latent user feature matrix (pattern matrix)
M = 1 * np.random.rand(r,n) # Latent movie feature matrix (coefficient matrix)
avg_movies = np.true_divide(R.sum(0),(R!=0).sum(0))
avg_movies[np.isnan(avg_movies)] = 0
M[0,:] = avg_movies # Set first row of Q to column vector of average ratings
E = np.eye(r,dtype=int) # rxr idendity matrix

train_errors = []
test_errors = []

# Repeat until convergence
while k < K:
    # Fix M and solve for U
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
        if (nui == 0): nui = 1 # remove zeros
    
        # Least squares solution
        Ai = np.add(np.dot(M, np.dot(np.diag(Ii), M.T)), np.multiply(l, np.multiply(nui, E))) # A_i = M_{I_i}M_{I_i}^T + ln_{u_i}E
        Vi = np.dot(M, np.dot(np.diag(Ii), R[i].T)) # V_i = M_{I_i}R^T(i,I_i)
        U[:,i] = np.linalg.solve(Ai,Vi).reshape(r)

    # Fix U and solve for M
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        if (nmj == 0): nmj = 1 # remove zeros

        # Least squares solution
        Aj = np.add(np.dot(U, np.dot(np.diag(Ij), U.T)), np.multiply(l, np.multiply(nmj, E))) $ A_j = # A_j = U_{I_j}U_{I_j}^T + ln_{m_j}E
        Vj = np.dot(U, np.dot(np.diag(Ij), R[:,j])) # V_j = U_{I_j}R(I_j,j)
        try:
            M[:,j] = np.linalg.solve(Aj,Vj).reshape(r)
        except: 
            print "singular matrix error: " + str(j + 1) # movie has no reviewers 
            continue

    train_rmse = rmse(I,R,M,U)
    test_rmse = rmse(I2,Rtst,M,U)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    print "[k: %d/%d] train-RMSE = %f  test-RMSE = %f" %(k+1, K, train_rmse, test_rmse)
    k = k + 1

format_output(I2,M,U)
print test_errors
