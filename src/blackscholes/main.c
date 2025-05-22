#include "blackscholes.h"
#include <ctimer.h>

int BlackScholes(int numOptions, 
                 OptionData* data,
                 int    * otype,
                 fptype * sptprice,
                 fptype * strike,
                 fptype * rate,
                 fptype * volatility,
                 fptype * otime,
                 fptype * prices);
 
int main (int argc, char **argv) {
  FILE *file;
  int i;
  int loopnum;
  fptype * buffer;
  int * buffer2;
  int rv;
  int numOptions = 0;

  if (argc != 3) {
    printf("Usage:\n\t%s <inputFile> <outputFile>\n", argv[0]);
    exit(1);
  }
  char *inputFile = argv[1];
  char *outputFile = argv[2];

  //Read input data from file
  file = fopen(inputFile, "r");
  if(file == NULL) {
    printf("ERROR: Unable to open file `%s'.\n", inputFile);
    exit(1);
  }
  rv = fscanf(file, "%i", &numOptions);
  if(rv != 1) {
    printf("ERROR: Unable to read from file `%s'.\n", inputFile);
    fclose(file);
    exit(1);
  }

  // alloc spaces for the option data
  OptionData* data = (OptionData*)malloc(numOptions*sizeof(OptionData));
  fptype* prices = (fptype*)malloc(numOptions*sizeof(fptype));
  for ( loopnum = 0; loopnum < numOptions; ++ loopnum ) {
    rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", 
                &data[loopnum].s, &data[loopnum].strike, 
                &data[loopnum].r, &data[loopnum].divq, 
                &data[loopnum].v, &data[loopnum].t, 
                &data[loopnum].OptionType, 
                &data[loopnum].divs, 
                &data[loopnum].DGrefval);
    if(rv != 9) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }
  }
  rv = fclose(file);
  if(rv != 0) {
    printf("ERROR: Unable to close file `%s'.\n", inputFile);
    exit(1);
  }
  printf("Num of Options: %d\n", numOptions);
  printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

  buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
  fptype * sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
  fptype * strike = sptprice + numOptions;
  fptype * rate = strike + numOptions;
  fptype * volatility = rate + numOptions;
  fptype * otime = volatility + numOptions;

  buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
  int *otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

  for (i=0; i<numOptions; i++) {
    otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
    sptprice[i]   = data[i].s;
    strike[i]     = data[i].strike;
    rate[i]       = data[i].r;
    volatility[i] = data[i].v;    
    otime[i]      = data[i].t;
  }

  printf("Size of data: %ld\n", numOptions * (sizeof(OptionData) + sizeof(int)));

  ctimer_t t;
  ctimer_start(&t);
  int nErrors = BlackScholes(numOptions, data, otype, sptprice, strike, rate, volatility, otime, prices);
  ctimer_stop(&t);
  ctimer_measure(&t);
  ctimer_print(t, "blackscholes");
  printf("NUmber of errors: %d\n", nErrors);

  //Write prices to output file
  file = fopen(outputFile, "w");
  if(file == NULL) {
    printf("ERROR: Unable to open file `%s'.\n", outputFile);
    exit(1);
  }
  rv = fprintf(file, "%i\n", numOptions);
  if(rv < 0) {
    printf("ERROR: Unable to write to file `%s'.\n", outputFile);
    fclose(file);
    exit(1);
  }
  for (i=0; i<numOptions; i++) {
    rv = fprintf(file, "%.18f\n", prices[i]);
    if (rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
      fclose(file);
      exit(1);
    }
  }
  rv = fclose(file);
  if(rv != 0) {
    printf("ERROR: Unable to close file `%s'.\n", outputFile);
    exit(1);
  }

#ifdef ERR_CHK
  printf("Num Errors: %d\n", numError);
#endif
  free(data);
  free(prices);

  return 0;
}

