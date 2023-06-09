# SpeedCodeBench

Benchmark Suite for SpeedCode (Software Performance Engineering Education via Coding Of Didactic Exercises).
This repository covers representative benchmarks in a wide range of application domains 
that can benefit from high performance parallel computing technology. 
The benchmarks are implemented using OpenMP and [OpenCilk](https://www.opencilk.org/)
for shared-memory multicore CPUs, and CUDA for GPUs.
This is useful if you are interested in comparing different aspects of each parallel language's approach to,
and ability to facilitate, parallel programming, including:
ease of use, compile time, performance and efficiency,
fine-grained control, and proneness to bugs such as deadlocks and race conditions.
It is also useful for evaluating newly designed and implemented CPU, GPU, accelerator, compiler, or operating system.

## Quick Start

Edit `env.sh` to let the libraries pointing to the right paths in your system, and then:

    $ source env.sh

Then just make in the root directory:

    $ make

Or go to each sub-directory, e.g. src/tc, and then make:

    $ cd src/tc; make

Binaries will be in the `bin` directory. 
For example, `tc_omp_base` is the OpenMP version of triangle counting on CPU, `tc_gpu_base` is the GPU version.

To run, go to each sub-directory, and then:

    $ ./run-test.sh

To find out commandline format by running executable without argument:

    $ cd ../../bin
    $ ./tc_omp_base

Run triangle counting with an undirected toy graph on CPU:

    $ ./tc_omp_base ../inputs/citeseer/graph
    
More graph datasets are available [here](https://www.dropbox.com/sh/i1jq1uwtkcd2qo0/AADJck_u3kx7FeSR5BvdrkqYa?dl=0).
You can find the expected outputs in the README of each benchmark [see here for triangle](https://github.com/chenxuhao/GraphAIBench/blob/master/src/triangle/README.md).

To control the number of threads, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]

## Benchmarks

### Kernels (Parallel Patterns)
- [x] Single-Precision A X Plus Y ([SAXPY](https://developer.nvidia.com/blog/six-ways-saxpy/))
- [x] Single Precision General Matrix Multiplication ([SGEMM](https://netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html)) 
- [x] Sparse Matrix-Vector Multiplication ([SpMV](https://en.wikipedia.org/wiki/Sparse_matrix%E2%80%93vector_multiplication))
- [x] Sparse Matrix Dense Matrix Multiplication ([SpMDM](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/))
- [x] [Stencil](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops)
- [x] [Convolution](https://en.wikipedia.org/wiki/Convolution) (CONV)
- [x] [Prefix Sum](https://en.wikipedia.org/wiki/Prefix_sum) a.k.a. Scan
- [x] [Histogram](https://en.wikipedia.org/wiki/Histogram)
- [x] [Reduction](https://en.wikipedia.org/wiki/Reduction_operator) (Sum / Maximum Finding / MinMax)
- [x] [Merge Sort](https://en.wikipedia.org/wiki/Merge_sort)
- [x] [Radix Sort](https://en.wikipedia.org/wiki/Radix_sort)
- [x] [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method) for solving linear systems $Ax=b$
- [x] [Breadth-First Search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS) / Graph Traversal
- [x] [Fast Fourier transform](https://people.sc.fsu.edu/~jburkardt/c_src/fft_openmp/fft_openmp.html) (FFT) for digital signal processing


|             |       Serial       |    OpenMP           |        Cilk        |      CUDA          |
|-------------|-------------------:|--------------------:|-------------------:|-------------------:|
|   SAXPY     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   SGEMM     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   SpMV      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   SpMDM     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Stencil   | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   CONV      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Scan      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Histo     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Reduce    | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Merge     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Radix     | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   Jacobi    | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   BFS       | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|   FFT       | :heavy_check_mark: |   |  | :heavy_check_mark: |

### Applications & Algorithms
- [x] Advanced Encryption Standard (AES), a specification for the encryption of electronic data
- [x] B+ tree (B+T) used in file systems and database systems
- [x] Barnes-Hut (BH) for N-Body simulation
- [x] Black-Scholes (BS), a differential equation to price options contracts in finance
- [x] CRC64 checksum (CRC), an error-detecting code used in digital networks and storage devices
- [x] Jaccard index (JI), a statistic used for gauging the similarity and diversity of sample sets
- [x] Haversine Distance (HD) for Geospatial Data Analysis
- [x] Symmetric Gauss-seidel Smoother (SymGS) for numerical linear algebra 
- [x] Inversek2j (IK2J) Inverse kinematics for 2-joint arm used in Robotics
- [x] Lattice Boltzmann methods (LBM) for Computational Fluid Dynamics (CFD)
- [x] Collaborative Filtering (CF), a Stochastic Gradient Descent (SGD) algorithm for recommender systems
- [x] K-means clustering (K-MEANS), a method of vector quantization for signal processing
- [x] Ray Tracing (RT) for 3D computer graphics
- [x] k-nearest neighbor (k-NN), a supervised learning method for classification and regression
- [x] Locality sensitive hashing (LSH) for finding approximate nearest neighbors
- [x] StreamCluster (SC) for online clustering of an input stream
- [x] DAPHNE points2image (P2I) for Automotive
- [x] PageRank (PR) for ranking web pages in a search engine
- [x] Triangle Counting (TC) for social network analysis
- [x] Betweenness Centrality (BC), a measure of centrality in a graph (from graph theory)
- [x] Connected Components (CC) (from graph theory)
- [x] Single-Source Shortest Paths (SSSP) finding the shortest paths (from graph theory)
- [x] Minimum Spanning Tree (MST) (from graph theory)
- [x] Vertex Coloring (VC) (from graph theory)

|             |       Serial       |    OpenMP           |        Cilk        |      CUDA          |
|-------------|-------------------:|--------------------:|-------------------:|-------------------:|
|     BH      | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|     BS      | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|     LBM     | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|     CF      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|     PR      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|     TC      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|     CC      | :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
|     BC      | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|    SSSP     | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|     MST     | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |
|     VC      | :heavy_check_mark: |  :heavy_check_mark: |  | :heavy_check_mark: |

## Sources

+ Parboil https://github.com/abduld/Parboil 
+ Rodinia https://github.com/yuhc/gpu-rodinia 
+ PARSEC https://github.com/bamos/parsec-benchmark 
+ SPLASH-2https://github.com/staceyson/splash2 
+ GAPBS https://github.com/sbeamer/gapbs 
+ Seven Dwarfs https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-183.pdf 
+ PBBS https://cmuparlay.github.io/pbbsbench/benchmarks/index.html 
+ GBBS https://paralg.github.io/gbbs/docs/introduction 
+ HeCBench https://github.com/zjin-lcf/HeCBench
+ GARDENIA https://github.com/chenxuhao/gardenia
+ GraphAIBench https://github.com/chenxuhao/GraphAIBench
+ AxBench http://axbench.org/
+ DAPHNE https://github.com/esa-tu-darmstadt/daphne-benchmark.git
+ Lonestar https://iss.oden.utexas.edu/?p=projects/galois/lonestar

## Classification of High Performance Computational Problems

1. Dense Linear Algebra,  Matrix multiply (e.g., SGEMM)
2. Sparse Linear Algebra, (e.g., SpMV / SpMM)
3. Spectral Methods, (e.g., FFT)
4. N-Body Methods, (e.g., Barnes-Hut)
5. Structured Grids, (e.g., LBM)
6. Unstructured Grids 
7. Monte Carlo 
8. Combinational Logic (e.g., encryption)
9. Graph traversal (e.g., Quicksort) 
10. Dynamic Programming 
11. Backtrack and Branch+Bound 
12. Construct Graphical Models 
13. Finite State Machine 

## Benchmark categories in application domians

1. Bioinformatics (e.g., all-pairs-distance)
2. Computer vision and image processing (e.g., Stencil, Convolution, aobench, sad, sobel, mriQ)
3. Cryptography (e.g., AES)
4. Data compression and reduction (e.g., Scan, bitpacking, histogram)
5. Data encoding, decoding, or verification (e.g., md5hash, crc64)
6. Finance (e.g., black-scholes)
7. Geographic information system (e.g., haversine)
8. Graph and Tree (e.g., BC, CC, TC, VC, MST, MIS, SSSP)
9. Language and kernel features (e.g., wordcount, saxpy)
10. Machine learning (e.g., CF, backprop, attention, kmeans, knn, page-rank, streamcluster, word2vec)
11. Math (e.g., sgemm, spmv, symgs, jaccard, jacobi, leukocyte, lud, tridiagonal solver, AMG, matrix-rotate)
12. Random number generation (e.g., rng-wallace, sobol)
13. Search (e.g., binary search, b+tree, BFS)
14. Signal processing (e.g., FFT)
15. Simulation (e.g., nbody, LBM, CFD, Delaunay Mesh Refinement, hotspot3D, heartwall, laplace3d, lavaMD, particlefilter, pathfinder, pns, tpacf, bspline-vgh, burger, minisweep, miniWeather, sph, testSNAP)
16. Sorting (e.g., quicksort, radixsort, mergesort, bitonic-sort)
17. Robotics (e.g., inversek2j)
18. Automotive (e.g., daphne)
