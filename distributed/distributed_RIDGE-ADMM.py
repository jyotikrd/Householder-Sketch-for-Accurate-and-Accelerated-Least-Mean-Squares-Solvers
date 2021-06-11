"""*****************************************************************************************
MIT License
Copyright (c) 2021 Jyotikrishna Dass and Rabi Mahapatra
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


"""####################################### NOTES ############################################
# - Please cite our paper when using the code: 
#       "Householder Sketch for Accurate and Accelerated Least-Mean-Squares Solvers" (ICML 2021)
#                          Jyotikrishna Dass and Rabi Mahapatra
#
#  Special thanks to Rengang Yang
##########################################################################################"""


# sets max threads before doing lapack
import mkl
mkl.set_num_threads(1)

import time, h5py, numpy as np, pandas as pd, sys
from numpy.linalg import norm
from mpi4py import MPI
from sklearn import linear_model

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# changable variables
lamb = 0.1          # lambda used in ridge regularization
zero_tol = 1e-10     # tolerace until set to zero, fpoint
rho = 1e6           # langrangian parameter
max_iter = 100000   # maximum admm iterations


def readInData(file, rank):
    with h5py.File(file,'r') as f:
        X = np.copy(f['data/X'+str(rank)][()])
        y = np.copy(f['data/y'+str(rank)][()])
    return X, y

def main():
    N = size

    # default
    inputFile = "data.h5"

    if(len(sys.argv) > 1):
        inputFile = sys.argv[1]

    X,  y = readInData(inputFile, rank)
    m, n = X.shape

    if rank==0:
        t_start = time.perf_counter()
    
    #initialize ADMM variables
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    r = np.zeros((n,1))

    send = np.zeros(3)
    recv = np.zeros(3)

    # cache XtX
    XtX = np.transpose(X)@X + rho * np.identity(n)
    clf = linear_model.Ridge(alpha=0.1, fit_intercept=False)
    clf.fit(X, y)

    '''
        ADMM solver loop, based on Boyd's C implementation: https://web.stanford.edu/~boyd/papers/admm/mpi/
    '''
    for k in range(max_iter):
        # u-update
        u+=(x-z)

        # x-update 
        x2 = np.linalg.solve(XtX, rho*(z-u))
        x = clf.coef_.reshape(n, 1) + x2

        w = x + u

        send[0] = (np.transpose(r) @ r)
        send[1] = (np.transpose(x) @ x)
        send[2] = (np.transpose(u) @ u) / (rho**2)
        
        comm.Allreduce(w, z, op=MPI.SUM)
        comm.Allreduce(send, recv, op=MPI.SUM)

        # z-update, ridge
        z = (rho*z)/(2+(N*rho))

        # more optimized admm loop
        if (np.sqrt(recv[0])) < 1e-6 and k>0:
            break
        
        # Compute residual
        r = x-z
            

    if rank==0:
        t_total = time.perf_counter()-t_start
        print("\nElapsed time is %.8f seconds"%t_total)
        print("Iterations:", k, "| Final r norm:", np.sqrt(recv[0]))


        x[np.abs(x) < zero_tol] = 0
        df_w = pd.DataFrame(x).transpose()
        print("Gotten Weights:")
        print(df_w.head())
        with open('admm_weights.csv', 'a+') as f:
            df_w.to_csv(f, header=False, index=False)


if __name__=='__main__':
    main()
