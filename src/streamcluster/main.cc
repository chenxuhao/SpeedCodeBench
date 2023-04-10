#include "stream.h"

int main(int argc, char **argv) {
  char *outfilename = new char[MAXNAMESIZE];
  char *infilename = new char[MAXNAMESIZE];
  long kmin, kmax, n, chunksize, clustersize;
  int dim;
  if (argc<10) {
    fprintf(stderr,"usage: %s k1 k2 d n chunksize clustersize infile outfile nthreads\n",
        argv[0]);
    fprintf(stderr,"  k1:          Min. number of centers allowed\n");
    fprintf(stderr,"  k2:          Max. number of centers allowed\n");
    fprintf(stderr,"  d:           Dimension of each data point\n");
    fprintf(stderr,"  n:           Number of data points\n");
    fprintf(stderr,"  chunksize:   Number of data points to handle per step\n");
    fprintf(stderr,"  clustersize: Maximum number of intermediate centers\n");
    fprintf(stderr,"  infile:      Input file (if n<=0)\n");
    fprintf(stderr,"  outfile:     Output file\n");
    fprintf(stderr,"  nthreads:    Number of threads to use\n");
    fprintf(stderr,"\n");
    fprintf(stderr, "if n > 0, points will be randomly generated instead of reading from infile.\n");
    exit(1);
  }
  kmin = atoi(argv[1]);
  kmax = atoi(argv[2]);
  dim = atoi(argv[3]);
  n = atoi(argv[4]);
  chunksize = atoi(argv[5]);
  clustersize = atoi(argv[6]);
  strcpy(infilename, argv[7]);
  strcpy(outfilename, argv[8]);
  int nthreads = atoi(argv[9]);
  omp_set_num_threads(nthreads);
  srand48(SEED);
  PStream* stream;
  if( n > 0 ) {
    stream = new SimStream(n);
  }
  else {
    stream = new FileStream(infilename);
  }
  double t1 = omp_get_wtime();
  streamCluster(stream, kmin, kmax, dim, chunksize, clustersize, outfilename);
  double t2 = omp_get_wtime();
  printf("time = %lf\n",t2-t1);
  delete stream;
#ifdef PROFILE
  printf("time pgain = %lf\n", time_gain);
  printf("time pgain_dist = %lf\n", time_gain_dist);
  printf("time pgain_init = %lf\n", time_gain_init);
  printf("time pselect = %lf\n", time_select_feasible);
  printf("time pspeedy = %lf\n", time_speedy);
  printf("time pshuffle = %lf\n", time_shuffle);
  printf("time localSearch = %lf\n", time_local_search);
#endif
  return 0;
}
