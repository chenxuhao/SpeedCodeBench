#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define UINT8_MAX 255

void get_histo_size(unsigned int * img, int img_width, int img_height, int *max_val, int *min_val) {
  int ii;
  int num_threads = omp_get_max_threads();
  printf("threas num = %d\n", num_threads);
  int max_temp[num_threads];
  int min_temp[num_threads];

  //calculate the temp min and max in parallel
  #pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    int min = img[0];
    int max = img[0];
    #pragma omp for
    for(ii = 1; ii < img_width*img_height; ii++) {
      if(img[ii]<min) {
        min = img[ii];
      }
      if(img[ii]>max) {
        max = img[ii];
      }
    }
    min_temp[tid] = min;
    max_temp[tid] = max;
  }
  
  *max_val=max_temp[0];
  *min_val=min_temp[0];

 //combine the results
  for(ii = 1; ii < num_threads; ii++) {
    if(*min_val>min_temp[ii])
      *min_val = min_temp[ii];
    if(*max_val<max_temp[ii])
      *max_val = max_temp[ii];
  }
}

void histogram(int n,
               unsigned int img_width, unsigned int img_height,
               unsigned int* image,
               unsigned int width, unsigned int height,
               unsigned char* histo) {
  //calculate the max and min value of image[]
  int max_val;
  int min_val;
  {
    // Write results into temp1 and temp2, then copy to max_val and min_val
    // This ensures that max_val and min_val are not address-taken
    int temp1;
    int temp2;
    get_histo_size(image, img_width, img_height, &temp1, &temp2);
    max_val = temp1;
    min_val = temp2;
  }
 
  int iter;
  for (iter = 0; iter < n; iter++){
    memset(histo, 0, height*width*sizeof(unsigned char));
    int num_threads = omp_get_max_threads();
    int num_histo =max_val-min_val+1;
    unsigned char * private_histo = (unsigned char*) calloc (num_threads*num_histo, sizeof(unsigned char));

    #pragma omp parallel
    {
    int i;
    int tid = omp_get_thread_num();
    int index = tid*num_histo;

    //initialize private_histo
    for(i=0;i<num_histo;i++)
      private_histo[index+i] = 0;
    #pragma omp barrier

    //accumulate the private histo
    #pragma omp for
    for(i = 0; i < img_width*img_height; i++) {
      private_histo[index+image[i]-min_val]++;
    }

    //combine the result into histo
    int t,j;
    #pragma omp for
    for(j=min_val;j<max_val+1;j++)
      for(t = 0; t < num_threads; t++) {
        unsigned char temp = histo[j];
        histo[j] += private_histo[t*num_histo+j-min_val];
        if (histo[j] < temp)
          histo[j] = UINT8_MAX;
      }
    }
  }
}
