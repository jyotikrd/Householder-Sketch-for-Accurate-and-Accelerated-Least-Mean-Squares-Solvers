==============================================================================

**Paper Title:** Householder Sketch for Accurate and Accelerated Least-Mean-Squares Solvers

**Authors:** Jyotikrishna Dass and Rabi Mahapatra 

**Affiliation:** Dept. of Computer Science and Engineering, Texas A&M University


Please cite our above ICMl 2021 when using the code
==============================================================================


Abstract: 
Least-Mean-Squares (\textsc{LMS}) solvers comprise a class of fundamental optimization problems such as linear regression, and regularized regressions such as Ridge, LASSO, and Elastic-Net. Data summarization techniques for big data generate summaries called coresets and sketches to speed up model learning under streaming and distributed settings. For example, \citep{nips2019} design a fast and accurate Caratheodory set on input data to boost the performance of existing \textsc{LMS} solvers. In retrospect, we explore classical Householder transformation as a candidate for sketching and accurately solving LMS problems. We find it to be a simpler, memory-efficient, and faster alternative that always existed to the above strong baseline. We also present a scalable algorithm based on construction of distributed Householder sketches to solve \textsc{LMS} problem across multiple worker nodes. We perform thorough empirical analysis with large synthetic and real datasets to evaluate the performance of Householder sketch and compare with \citep{nips2019}. Our results show Householder sketch speeds up existing \textsc{LMS} solvers in the scikit-learn library up to $100$x-$400$x. Also, it is $10$x-$100$x faster than the above baseline with similar numerical stability. The distributed algorithm demonstrates linear scalability with a near-negligible communication overhead.



ORGANIZATION:
- The code files are organized based on Sequential and Distributed implementation
- Settings are provided in the main paper, Section 4 and in Figure captions (wherever necessary)

- Sequential:
	-- Use Google Colab to run the file
	-- Refer Figure 1(a-l) and Figure 4(c) in the main paper (plots with bigger fonts are provided in supplementary pdf)
	-- Default is running on Synthetic dataset created within the code
	-- You may choose to run code on real datasets provided by uncommenting the corresponding lines in the code
	

- Distributed: 
	-- Was run on a university computing cluster with specs noted in Section 4(i) in the main paper
	-- Refer Figures 2, 3 and 4(a-b) in the main paper (plots with bigger fonts are provided in supplementary pdf)
	-- LMS is RIDGE solver for experiments. Easly applicable to other LMS solvers by changing the solver name 
	
	-- partition_h5.py to create a synthetic dataset with p partitions. 
		--- python3 partition_h5.py -h  // prints how to use the file
		--- Example run:  python3 partition_h5.py -p 2 -sd 2000,4 
			// creates data.h5 file comprising p=2 partitions each with 2000 x 4 data size
			
	-- qr.py comprises helper functions for the householder QR and implicit multiplication of Qc/Q^Tc. This will be imported into the next file
	
	-- distributed_RIDGE-QR.py to compute distributed householder QR via MPI on partitioned datasets in data.h5 above and run Ridge regression solver. 
		--- pip install mpi4py // Install mpi4py using 
		--- Example run: mpiexec -n 2 python3 distributed_RIDGE-QR.py // here,  2 (for example) workers.

	
