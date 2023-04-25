#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

typedef uint32_t data_t;
typedef uint32_t index_t;

int* prefix_sum(int* x , int n);
void par_counting_rank(int *f , int n, int r, int *s);
int extract_bit_segment(int a, int k, int l) {
  return (((1 << l) - 1) & (a >> k));
}

extern "C"
void radix_sort(unsigned int* outdata,
                unsigned int* const indata,
                unsigned int n) {
  int b = 32;
  int *f = new int[n];
  int *s = new int[n];

  //TODO fix it to number of processing elements
  int p = 68;
  int q;
  int r = ceil(log(ceil(n/ (p * log(n)))));
  for(int k = 0; k < b ; k = k+r) {
    q = (k + r <= b) ? r : b-k;
    cilk_for (int i = 0; i < n; i++) {
      f[i] = extract_bit_segment(indata[i],k,q);
    }
    par_counting_rank(f,n,q,s);
    cilk_for (int i = 0; i < n; i++) {
      outdata[s[i]-1] = indata[i];
    }
    cilk_for (int i = 0; i < n; i++) {
      indata[i] = outdata[i];
    }
  }
  delete []f;
  delete []s;
}

int* prefix_sum(int* x , int n) {
  int* s = new int[n];
  if (n==1) {
    s[0] = x[0];
  } else {
    int *z ,*y= new int [n/2];
    cilk_for(int i=0;i<n/2;i++) {
      y[i]=x[2*i]+x[2*i+1];           
    }
    z = prefix_sum(y,n/2);
    s[0] = x[0];
    cilk_for (int i=1;i<n;i++) {
      if (i%2 != 0) {
        s[i]=z[i/2];
      } else {
        s[i]=z[(i-1)/2] + x[i];
      }
    }
  }
  return s;
}

void par_counting_rank(int *f , int n, int r, int *s) {
  //TODO fix it to number of processing elements
  int p =68;
  int d = pow(2,r);

  int *js  = new int [p];
  int *je  = new int [p];
  int *ofs = new int [p];

  int** s1 = new int*[d];
  int** s2 = new int*[d];
  for(int i = 0; i < d; ++i) {
    s1[i] = new int[p];
    s2[i] = new int[p];
  }
  int **s1_ps = new int*[d];
  for (int i = 0; i < d; ++i) {
    s1_ps[i] = new int[p];
  }

  cilk_for (int i=0;i<p;i++) {
    for(int j=0;j<=d-1;j++) {
      s1[j][i]=0;
    }
    js[i] = (i) * ceil((n/p));
    je[i] = (i+1<p) ? (i+1) * (ceil(n/p)) - 1:n-1;

    for(int j=js[i]; j<=je[i]; j++){
      s1 [f[j]][i]= s1 [f[j]][i] + 1;
    }
  }
  int* send = new int[p];
  for (int j =0; j<= d-1; j++){
    for (int i = 0; i < p; ++i) {
      send[i] = s1[j][i];
    }
    int* ret = prefix_sum(send,p);
    for (int i = 0; i < p; ++i) {
      s1_ps[j][i] = ret[i];
    }
  }
  cilk_for (int i=0;i<p;i++) {
    ofs[i] = 1;
    for (int j=0; j<=d-1; j++) {
      s2[j][i] = (i == 0) ? ofs[i] : (ofs [i] + s1_ps[j][i-1]);
      ofs[i] = ofs[i] + s1_ps[j][p-1]; 
    }
    for (int j =js[i]; j<=je[i]; j++) {
      s[j] = s2[f[j]][i];
      s2[f[j]][i] = s2[f[j]][i]+1;
    }
  }
}

