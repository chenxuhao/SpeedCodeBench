#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>
#include <omp.h>

using namespace std;

#define MAXNAMESIZE 1024 // max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 // number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 3 // iterate ITER* k log k times; ITER >= 1

//#define PRINTINFO //comment this out to disable output
//#define PROFILE // comment this out to disable instrumentation code
//#define ENABLE_THREADS  // comment this out to disable threads
//#define INSERT_WASTE //uncomment this to insert waste computation into dist function

#define CACHE_LINE 512 // cache line in byte

/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
  float weight;
  float *coord;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
  long num; /* number of points; may not be N if this is a sample */
  int dim;  /* dimensionality */
  Point *p; /* the array itself */
} Points;

struct pkmedian_arg_t {
  Points* points;
  long kmin;
  long kmax;
  long* kfinal;
};

// instrumentation code
#ifdef PROFILE
static double time_local_search;
static double time_speedy;
static double time_select_feasible;
static double time_gain;
static double time_shuffle;
static double time_gain_dist;
static double time_gain_init;
#endif 

class PStream {
public:
  virtual size_t read( float* dest, int dim, int num ) = 0;
  virtual int ferror() = 0;
  virtual int feof() = 0;
  virtual ~PStream() {
  }
};

//synthetic stream
class SimStream : public PStream {
public:
  SimStream(long n_ ) {
    n = n_;
  }
  size_t read( float* dest, int dim, int num ) {
    size_t count = 0;
    for( int i = 0; i < num && n > 0; i++ ) {
      for( int k = 0; k < dim; k++ ) {
	dest[i*dim + k] = lrand48()/(float)INT_MAX;
      }
      n--;
      count++;
    }
    return count;
  }
  int ferror() {
    return 0;
  }
  int feof() {
    return n <= 0;
  }
  ~SimStream() { 
  }
private:
  long n;
};

class FileStream : public PStream {
public:
  FileStream(char* filename) {
    fp = fopen( filename, "rb");
    if( fp == NULL ) {
      fprintf(stderr,"error opening file %s\n.",filename);
      exit(1);
    }
  }
  size_t read( float* dest, int dim, int num ) {
    return std::fread(dest, sizeof(float)*dim, num, fp); 
  }
  int ferror() {
    return std::ferror(fp);
  }
  int feof() {
    return std::feof(fp);
  }
  ~FileStream() {
    printf("closing file stream\n");
    fclose(fp);
  }
private:
  FILE* fp;
};

void streamCluster( PStream* stream, long kmin, long kmax, int dim,
                    long chunksize, long centersize, char* outfile );
 
inline double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return (double)t.tv_sec+t.tv_usec*1e-6;
}

#ifdef INSERT_WASTE
inline double waste(double s) {
  for( int i =0 ; i< 4; i++ ) {
    s += pow(s,0.78);
  }
  return s;
}
#endif

/* compute Euclidean distance squared between two points */
inline float dist(Point p1, Point p2, int dim) {
  int i;
  float result=0.0;
  for (i=0;i<dim;i++)
    result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);
#ifdef INSERT_WASTE
  double s = waste(result);
  result += s;
  result -= s;
#endif
  return(result);
}

