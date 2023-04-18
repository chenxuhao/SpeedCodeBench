# SpeedCodeBench

Benchmark Suite for SpeedCode (Software Performance Engineering Education via Coding Of Didactic Exercises).
This repository covers representative benchmarks in a wide range of application domains 
that can benefit from high performance parallel computing technology. 


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

- [x] AES Encryption
- [x] Black-Scholes a differential equation to price options contracts
- [x] CRC64 checksum
- [x] Stencil
- [x] Jaccard index, a.k.a, Jaccard similarity coefficient
- [x] Haversine distance for Geospatial Data Analysis
- [x] Saxpy: Single-Precision A X Plus Y
- [x] Single Precision General Matrix Multiplication (SGEMM) 
- [x] Sparse Matrix-Vector Multiplication (SpMV)
- [x] Symmetric Gauss-seidel Smoother (SymGS) 
- [x] Histogram
- [x] Inversek2j for Robotics
- [x] Reduction (Sum / Maximum Finding / MinMax)
- [x] Convolution
- [x] Merge / Merge Sort
- [x] Radix Sort
- [x] Scan a.k.a. Prefix Sum
- [x] Barnes-Hut for N-Body simulation
- [x] Lattice Boltzmann methods (LBM) for Computational Fluid Dynamics (CFD)
- [x] Fast Fourier transform (FFT)
- [x] Betweenness Centrality (BC)
- [x] Breadth-First Search (BFS)
- [x] Connected Components (CC)
- [x] PageRank (PR) 
- [x] Single-Source Shortest Paths (SSSP)
- [x] Triangle Counting (TC)
- [x] Minimum Spanning Tree (MST) 
- [x] Vertex Coloring (VC)
- [x] Collaborative Filtering (CF) for Stochastic Gradient Descent (SGD)
- [x] B+ tree
- [x] K-means clustering
- [x] Ray Tracing (RT)
- [x] k-nearest neighbor (k-NN)
- [x] Locality sensitive hashing (LSH) for finding approximate nearest neighbors
- [x] StreamCluster (SC) for online clustering of an input stream
- [x] DAPHNE points2image for Automotive

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
