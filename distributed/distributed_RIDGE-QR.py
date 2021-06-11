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
from sklearn.linear_model import Ridge


# changeable variables
lamb = 0.1          # lambda used in ridge regularization
zero_tol = 1e-10     # tolerace until set to zero, fpoint


def readInData(file, rank):
    # https://onestopdataanalysis.com/hdf5-in-python-h5py/
    with h5py.File(file,'r') as f:
        X = np.copy(f['data/X'+str(rank)][()])
        y = np.copy(f['data/y'+str(rank)][()])
    return X, y

def main():
	global comm, rank, size
	# Default file
	folder = "data.h5"

	# commandline args
	if(len(sys.argv) > 1):
		folder = sys.argv[1]

	X_i, b_i = readInData(folder, rank)
	n, k = X_i.shape[0], X_i.shape[1]

	 # Starts the timer.
	if(rank == 0):
		t_start = time.perf_counter()

	#############################
	##### Slice 1: Local QR #####
	#############################
	H_i, tau_i, R_i = qr(X_i)
	if(rank == 0):
		t_slice1 = time.perf_counter() - t_start
		#print("Time in Seconds, Local QR:", t_slice1)

	#########################################
	##### Slice 2: local b_hat_partials #####
	#########################################
	# Computing: b_hat_partial_i = (Q_i^T) x b_i
	#########################################
 
	b_hat_partial_i = multiplyQC(H_i, tau_i, b_i, trans="T") 
	if(rank == 0):
		t_slice2 = time.perf_counter() - t_start - t_slice1
		#print("Time in Seconds, local b_hat_partials:", t_slice2)

	#################################################
	##### Slice 3: gather R, top k rows of each #####
	#################################################
	R_stacked = comm.gather(R_i[:k], root=0)
	if(rank == 0):
		t_slice3 = time.perf_counter() - t_start - (t_slice1 + t_slice2)
		#print("Time in Seconds, gather R:", t_slice3)

	########################################################################
	######### Slice 4: gather b_hat_partial, top k rows of each ############
	# Communicate: b_hat_partial = stack(b_hat_partial_i) from all workers, 
	# i.e., effectively perform, b_hat_partial = diag(Q_1,..,Q_p)^T x b
	########################################################################
	b_hat_partial = comm.gather(b_hat_partial_i[:k], root=0)
	if(rank == 0):
		t_slice4 = time.perf_counter() - t_start - (t_slice1 + t_slice2 + t_slice3)
		#print("Time in Seconds, gather b_hat:", t_slice4)

	w = None
	if(rank == 0):
		R_stacked = np.concatenate(R_stacked)
		b_hat_partial = np.concatenate(b_hat_partial)
		
		###################################################
		##### Slice 5: calculates Q_m and R at Master #####
		###################################################
		H_m, tau_m, R = qr(R_stacked)
		t_slice5 = time.perf_counter() - t_start - (t_slice1 + t_slice2 + t_slice3 + t_slice4)
		#print("Time in Seconds, master QR:", t_slice5)

		####################################################################################
		############################# Slice 6: calculate, b_hat ############################
		# Compute, b_hat = Q_m^T x b_hat_partial = Q_m^T x diag(Q_1,..,Q_p)^T x b = Q^T x b 
		###################################################################################
		b_hat = multiplyQC(H_m, tau_m, b_hat_partial, trans="T")
		t_slice6 = time.perf_counter() - t_start - (t_slice1 + t_slice2 + t_slice3 + t_slice4 + t_slice5)
		#print("Time in Seconds, b_hat:", t_slice6)
		
		###############################################
		##### slice 7: fitting/training the model #####
		###############################################
		clf = Ridge(alpha=lamb, fit_intercept=False)
		clf.fit(R, b_hat.ravel()[:k])
		w = clf.coef_
		t_slice7 = time.perf_counter() - t_start - (t_slice1 + t_slice2 + t_slice3 + t_slice4 + t_slice5 + t_slice6)
		#print("Time in Seconds, fit model:", t_slice7)

	#############################################
	##### slice 8: Broadcast weights to all #####
	#############################################
	w = comm.bcast(w, root=0)
	if(rank == 0):
		t_slice8 = time.perf_counter() - t_start - (t_slice1 + t_slice2 + t_slice3 + t_slice4 + t_slice5 + t_slice6 + t_slice7)
		#print("Time in Seconds, Broadcast w:", t_slice8)

		# outputs the values
		#if(rank == 0):
		# time
		t_total = time.perf_counter() - t_start
		print("Time in Seconds:", t_total)

		print("###### Time Breakdowns ######")
		print("Time in Seconds, Local QR:", t_slice1)
		print("Time in Seconds, Local b_hat_partials:", t_slice2)
		print("Time in Seconds, gather R:", t_slice3)
		print("Time in Seconds, gather b_hat:", t_slice4)
		print("Time in Seconds, master QR:", t_slice5)
		print("Time in Seconds, b_hat:", t_slice6)
		print("Time in Seconds, fit model:", t_slice7)
		print("Time in Seconds, Broadcast w:", t_slice8)

		w[np.abs(w) < zero_tol] = 0.0
		#prints the weights
		df_w = pd.DataFrame(w).transpose()
		
		print("Gotten Weights:")
		pd.set_option('display.max_columns', None) 
		print(df_w.head())
		with open('ridgeqr_weights.csv', 'a+') as f:
			df_w.to_csv(f, header=False, index=False)


if __name__ == "__main__":
	main()
