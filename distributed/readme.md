**Distributed Implementation:**

	-- Was run on a university computing cluster with specs noted in Section 4(i) in the main paper
  
	-- Refer Figures 2, 3 and 4(a-b) in the main paper (plots with bigger fonts are provided in supplementary pdf)
  
	-- LMS is RIDGE solver for experiments. Easly applicable to other LMS solvers by changing the solver name 
	
	-- partition_h5.py to create a synthetic dataset with p partitions. 
  
		--- python3 partition_h5.py -h  // prints how to use the file
    
		--- Example run:  python3 partition_h5.py -p 2 -sd 2000,4 
    
			// creates data.h5 file comprising p=2 partitions each with 2000 x 4 data size
      
			
	-- qr.py comprises helper functions for the householder QR and implicit multiplication of Qc/Q^Tc. This will be imported into the next file
 
	
	-- distributed_RIDGE-QR.py to compute distributed householder QR via MPI on partitioned datasets in data.h5 above and run Ridge regression solver. 
  
		--- pip install mpi4py // Install mpi4py
    
		--- Example run: mpiexec -n 2 python3 distributed_RIDGE-QR.py // here,  2 (for example) workers.
