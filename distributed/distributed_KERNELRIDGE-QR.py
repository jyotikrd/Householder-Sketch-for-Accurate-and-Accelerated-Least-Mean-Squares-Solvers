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


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# sets max threads before doing lapack
import mkl
mkl.set_num_threads(1)
import pandas as pd, numpy as np, time, sys, scipy, h5py
from qr import qr, multiplyQC
from sklearn.kernel_ridge import KernelRidge

# changable variables
lamb = 0.1          # lambda used in ridge regularization
zero_tol = 1e-10     # tolerace until set to zero, fpoint

def readInData(file, rank):
    with h5py.File(file,'r') as f:
        X = np.copy(f['data/X'+str(rank)][()])
        y = np.copy(f['data/y'+str(rank)][()])
    return X, y

def main():
    global comm, rank, size
    # preset variables
    folder = "data.h5"

    # commandline args
    if(len(sys.argv) > 1):
        folder = sys.argv[1]

    # Slice 0: Individual data loading
    X_i, b_i = readInData(folder, rank)
    n, k = X_i.shape[0], X_i.shape[1]

    # Starts the timer.
    if(rank == 0):
        t_start = time.perf_counter()

    # Slice 1: Local QRs
    H_i, tau_i, R_i = qr(X_i)

    # Slice 2: local b_hat_partials
    b_hat_partial_i = multiplyQC(H_i, tau_i, b_i, trans="T")

    # Slice 3: gather R, top k rows of each
    R_stacked = comm.gather(R_i[:k], root=0)

    # Slice 4: gather b_hat_partial, top k rows of each
    b_hat_partial = comm.gather(b_hat_partial_i[:k], root=0)

    w = None
    if(rank == 0):
        R_stacked = np.concatenate(R_stacked)
        b_hat_partial = np.concatenate(b_hat_partial)

        # Slice 5: calculates Q_g and R
        H_g, tau_g, R = qr(R_stacked)

        # Slice 6: calculates b_hat
        b_hat = multiplyQC(H_g, tau_g, b_hat_partial, trans="T")

        # slice 7: fitting/training the model
        clf = KernelRidge(alpha=lamb, kernel="linear")
        clf.fit(R, b_hat.ravel()[:k])
        w = np.transpose(R) @ clf.dual_coef_


    # slice 8: Broadcast weights to all.
    w = comm.bcast(w, root=0)

    # outputs the values
    if(rank == 0):
        t_total = time.perf_counter() - t_start
        print("Time in Seconds:", t_total)


        w[np.abs(w) < zero_tol] = 0
        #prints the weights
        df_w = pd.DataFrame(w).transpose()
        print("Gotten Weights:")
        print(df_w.head())

        # also outputs weights to csv
        with open('kernelridgeqr_weights.csv', 'a+') as f:
            df_w.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    main()