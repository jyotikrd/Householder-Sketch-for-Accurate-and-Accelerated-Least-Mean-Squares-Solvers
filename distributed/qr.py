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



import scipy.linalg.lapack as LAPACK
import pandas as pd, numpy as np, sys, scipy
'''
this is a special qr decomposition function that returns
householder reflectors and R
    returns H:      Elementary Reflectors
            tau:    Scaling factor (?) 
            R:      Upper triangular matrix
    for tau, please read:
    https://scicomp.stackexchange.com/questions/30989/what-is-the-reason-that-lapack-uses-tau-in-qr-decomposition-instead-of-norma
'''
def qr(X):
    output = LAPACK.dgeqrf(X)
    H = output[0]
    tau = output[1]
    R = np.triu(H[:X.shape[1], :])
    return H, tau, R


"""
Multiplies a matrix with Q's household reflectors. Calculates Q @ M without Q
Uses LAPACK routines exposed in SciPy

https://software.intel.com/en-us/mkl-developer-reference-c-ormqr#0AB81CED-C4E5-4E1B-9CE1-EE5BE66828EE
Had to do a lot of digging in scipy, but basically, there
are two LAPACK routines that match the ormqr that we need:
sormqr and dormqr, AFAIK, they do the exact same thing, so
I'm going to use dormqr, since scipy does as well in qr_multiply

d = float64, s = float32. so we're using d
"""
def multiplyQC(H, tau, C, side='L', trans='N'):
  output = LAPACK.dormqr(side=side, trans=trans, a=H, tau=tau, c=C, lwork=C.shape[0])
  out_clean = np.asarray_chkfinite(output[:-2])[0]
  return out_clean