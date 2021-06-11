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



import sys, os, shutil, csv, pandas as pd, time,scipy.linalg, numpy as np, math, argparse
import pdb,h5py,os

# https://onestopdataanalysis.com/hdf5-in-python-h5py/

"""
    Flag: -sd 10,5,-100,100,0 will means n=10, d=5, min=-100, max=100, seed=0
    Creates a random synthetic dataset of n samples with d features
        - n (int): Number of samples/rows
        - d (int): Number of features
        - a_min (int): Smallest possible value  (default -100)
        - a_max (int): largest possible value   (default 100)
        - seed (int): seed value, for reproducibility (default 0)
"""
def synthetic_data(n, d, a_min=-100, a_max=100, seed=0):
    np.random.seed(0)
    data = np.random.uniform(low=a_min, high=a_max, size=[n, d+1])  # uniform random
    df_data = pd.DataFrame(data)
    return df_data

def main():
    # ======== Creates the argument parser ================
    parser = argparse.ArgumentParser(description="Partitions Dataset")
    parser.add_argument("-p", "--partitions", type=int, help="Number of partitions, numbered from [0-p)")
    # dataset. 
    groupData = parser.add_mutually_exclusive_group()
    groupData.add_argument("-f", "--file", type=str, help="Use prepared dataset file, csv, as prepared.")
    groupData.add_argument("-sd", "--syntheticdata", type=str, help="Synthetic dataset, takes 2 or 5 arguments. First two are always n and d. (Check Code documentation)")
    # truncate
    parser.add_argument("-ml", "--maxlines", type=int, help="Most number of lines allowed.")
    # output
    parser.add_argument('-o', '--outfolder', type=str, default="data.h5", help="Name of timing output folder. (Default: 'data.h5')")
    args = parser.parse_args()
    # ======================================================

    # Checking -> partitions and data defined
    if(args.partitions == None):
        print("Please specify number of nodes.")
        exit(0)
    if(args.file == None and args.syntheticdata == None):
        print("Please specify either file or synthetic data.")
        exit(0)

    # synthetic dataset.
    if(args.syntheticdata != None):
        synth_args_str = args.syntheticdata.split(",")
        synth_args = [int(i) for i in synth_args_str]
        if(len(synth_args) == 2):
            data_df = synthetic_data(synth_args[0], synth_args[1])
        elif (len(synth_args) == 5):
            data_df = synthetic_data(synth_args[0], synth_args[1], synth_args[2], synth_args[3], synth_args[4])
        else:
            print("Synthetic data takes 2 or 5 arguments. Please check readme.")
            exit(0)
    # given csv file
    elif(args.file != None):
        print("Using dataset csv:", args.file)
        print("Make sure the first column is the targets, and that there are no headers!")
        if(args.maxlines != None):
            data_df = pd.read_csv(args.file, header=None, nrows=args.maxlines)
        else:
            data_df = pd.read_csv(args.file, header=None)

    # centers data
    data_df = data_df - data_df.mean(axis=0)


    # Adds 1 for intercept
    data_df.insert(1, 'y', 1)

    # writes out data to h5 file
    with h5py.File(args.outfolder,'w') as f:
        g = f.create_group('data')

        partition_size = data_df.shape[0]//args.partitions
        for x in range(args.partitions):
            g.create_dataset(name='X'+str(x) ,data=data_df.iloc[partition_size*x:partition_size*(x+1),1:],compression="gzip")
            g.create_dataset(name='y'+str(x) ,data=data_df.iloc[partition_size*x:partition_size*(x+1),0],compression="gzip")
    print("Data Partitioned!")

if __name__ == "__main__":
    main()
