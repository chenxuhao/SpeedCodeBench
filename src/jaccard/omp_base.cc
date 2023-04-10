#include <stdio.h>
#include <algorithm> 
#include <iostream> 
#include <vector> 
#include <chrono>
#include "graph.h"
#include "scan.h"
#include "platform_atomics.h"

using namespace std;

template <bool weighted, typename T>
void fill_weights(eidType e, T* w, const T value) {
  for (eidType j = 0; j<e; j++)
    w[j] = weighted ? (T)(j+1)/e : value; 
}

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
void jaccard_row_sum(vidType n, eidType* csrPtr, vidType* csrInd, T* weight_j, T* work) {
  for (vidType row = 0; row < n; row++) {
    auto start = csrPtr[row];
    auto end   = csrPtr[row+1];
    auto length= size_t(end-start);
    if (weighted) {
      T sum = parallel_prefix_sum<vidType,T>(length, csrInd + start, weight_j); 
      work[row] = sum;
    } else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Note the number of columns is constrained by the number of rows
template<bool weighted, typename T>
void jaccard_is(vidType n, eidType e, eidType* csrPtr, vidType* csrInd, 
                T* weight_j, T* work, T* weight_i, T* weight_s) {
  for (vidType row = 0; row < n; row++) {
    for (int j = csrPtr[row]; j < csrPtr[row+1]; j++) {
      int col = csrInd[j];
      //find which row has least elements (and call it reference row)
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref= (Ni < Nj) ? row : col;
      int cur= (Ni < Nj) ? col : row;

      //compute new sum weights
      weight_s[j] = work[row] + work[col];

      //compute new intersection weights 
      //search for the element with the same column index in the reference row
      for (int i = csrPtr[ref]; i < csrPtr[ref+1]; i++) {
        int match  =-1;
        int ref_col = csrInd[i];
        T ref_val = weighted ? weight_j[ref_col] : (T)1.0;

        //binary search (column indices are sorted within each row)
        int left = csrPtr[cur]; 
        int right= csrPtr[cur+1]-1; 
        while(left <= right){
          int middle = (left+right)>>1; 
          int cur_col= csrInd[middle];
          if (cur_col > ref_col) {
            right=middle-1;
          }
          else if (cur_col < ref_col) {
            left=middle+1;
          }
          else {
            match = middle; 
            break; 
          }
        }            

        //if the element with the same column index in the reference row has been found
        if (match != -1) {
          //fetch_and_add(weight_i[j], ref_val);
          #pragma omp critical
          weight_i[j] += ref_val;
        }
      }
    }
  }
}

template<bool weighted, typename T>
void jaccard_jw(eidType e, T* csrVal, const T gamma,  
                T* weight_i, T* weight_s, T* weight_j) {
  for (int j = 0; j < e; j++) {
    T Wi =  weight_i[j];
    T Ws =  weight_s[j];
    weight_j[j] = (gamma*csrVal[j])* (Wi/(Ws-Wi));
  }
}

template <bool weighted, typename T>
void jaccard_weight (GraphF &g, const int iteration) {
  const T gamma = (T)0.46;  // arbitrary
  auto n = g.V();
  auto e = g.E();
  auto csrPtr = g.rowptr();
  auto csrInd = g.colidx();
  auto csrVal = g.getElabelPtr();
  T* weight_i = (T*) malloc (sizeof(T) * e);
  T* weight_s = (T*) malloc (sizeof(T) * e);
  T* weight_j = (T*) malloc (sizeof(T) * e);
  T* work = (T*) malloc (sizeof(T) * n);
  double start_time = omp_get_wtime();
  for (int i = 0; i < iteration; i++) {
    fill_weights<weighted, T>(e, weight_j, (T)1.0);
    // initialize volume of intersections
    fill_weights<false, T>(e, weight_i, (T)0.0);
    jaccard_row_sum<weighted,T>(n, csrPtr, csrInd, weight_j, work);
    // this is the hotspot
    jaccard_is<weighted,T>(n, e, csrPtr, csrInd, weight_j, work, weight_i, weight_s);
    // compute jaccard weights
    jaccard_jw<weighted,T>(e, csrVal, gamma, weight_i, weight_s, weight_j);
  }
  double end_time = omp_get_wtime();
  std::cout << "Average execution time of kernels: " << ((end_time-start_time) * 1e-9f) / iteration << " (s)\n";

#ifdef DEBUG
  // verify using known values when weighted is true
  float error; 

  if (weighted)
    error = std::fabs(weight_j[0] - 0.306667) +
      std::fabs(weight_j[1] - 0.000000) +
      std::fabs(weight_j[2] - 3.680000) +
      std::fabs(weight_j[3] - 1.380000) +
      std::fabs(weight_j[4] - 0.788571) +
      std::fabs(weight_j[5] - 0.460000);
  else
    error = std::fabs(weight_j[0] - 0.230000) +
      std::fabs(weight_j[1] - 0.000000) +
      std::fabs(weight_j[2] - 3.680000) +
      std::fabs(weight_j[3] - 1.380000) +
      std::fabs(weight_j[4] - 0.920000) +
      std::fabs(weight_j[5] - 0.460000);

  if (error > 1e-5) {
    for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
    printf("FAIL");
  } else {
    printf("PASS");
  }
  printf("\n");
#endif
  free(weight_j);
  free(weight_i);
  free(weight_s);
  free(work);
}

template void jaccard_weight<true,float>(GraphF &g, const int iteration);
template void jaccard_weight<false,float>(GraphF &g, const int iteration);
