#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <chrono>

#include "SDKBitMap.h"
#include "aes.h"

// utils
void convertGrayToGray(const uchar4 *pixels, uchar *gray, const int height, const int width);
void convertColorToGray(const uchar4 *pixels, uchar *gray, const int height, const int width);
void createRoundKey(uchar * eKey, uchar * rKey);
void keyExpansion(uchar * key, uchar * expandedKey,
                  unsigned int keySize, unsigned int explandedKeySize);

// verifier
void reference(uchar * output, uchar * input, uchar * rKey, unsigned int explandedKeySize, unsigned int width,
               unsigned int height, bool inverse, unsigned int rounds, unsigned int keySize);

// kernels
void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                const uint     width , 
                const uint     height , 
                const uint     rounds );
void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                const  uint    width , 
                const  uint    height , 
                const  uint    rounds);

int main(int argc, char * argv[]) {
  if (argc != 4) {
    printf("Usage: %s <iterations> <0 or 1> <path to bitmap image file>\n", argv[0]);
    printf("0=encrypt, 1=decrypt\n");
    return 1;
  }
  const unsigned int keySizeBits = 128;
  const unsigned int rounds = 10;
  const unsigned int seed = 123;
  const int iterations = atoi(argv[1]);
  const bool decrypt = atoi(argv[2]);
  const char* filePath = argv[3];
  SDKBitMap image;
  image.load(filePath);
  const int width  = image.getWidth();
  const int height = image.getHeight();

  /* check condition for the bitmap to be initialized */
  if (width <= 0 || height <= 0) return 1;
  std::cout << "Image width and height: " 
            << width << " " << height << std::endl;
  uchar4 *pixels = image.getPixels();
  unsigned int sizeBytes = width*height*sizeof(uchar);
  uchar *input = (uchar*)malloc(sizeBytes); 

  /* initialize the input array, do NOTHING but assignment when decrypt*/
  if (decrypt)
    convertGrayToGray(pixels, input, height, width);
  else
    convertColorToGray(pixels, input, height, width);
  unsigned int keySize = keySizeBits/8; // 1 Byte = 8 bits
  unsigned int keySizeBytes = keySize*sizeof(uchar);
  uchar *key = (uchar*)malloc(keySizeBytes);
  fillRandom<uchar>(key, keySize, 1, 0, 255, seed); 

  // expand the key
  unsigned int explandedKeySize = (rounds+1)*keySize;
  uchar *expandedKey = (uchar*)malloc(explandedKeySize*sizeof(uchar));
  uchar *roundKey    = (uchar*)malloc(explandedKeySize*sizeof(uchar));
  keyExpansion(key, expandedKey, keySize, explandedKeySize);
  for(unsigned int i = 0; i < rounds+1; ++i) {
    createRoundKey(expandedKey + keySize*i, roundKey + keySize*i);
  }

  // save device result
  uchar* output = (uchar*)malloc(sizeBytes);
  std::cout << "Executing kernel for " << iterations 
            << " iterations" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  auto start = std::chrono::steady_clock::now();
  for(int i = 0; i < iterations; i++) {
    if (decrypt) 
      AESDecrypt(
        (uchar4*)output,
        (uchar4*)input,
        (uchar4*)roundKey,
        rsbox,
        width, height, rounds);
    else
      AESEncrypt(
        (uchar4*)output,
        (uchar4*)input,
        (uchar4*)roundKey,
        sbox,
        width, height, rounds);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / iterations << " (s)\n";

  // Verify
  uchar *verificationOutput = (uchar *) malloc(width*height*sizeof(uchar));
  reference(verificationOutput, input, roundKey, explandedKeySize, 
      width, height, decrypt, rounds, keySize);
  /* compare the results and see if they match */
  if(memcmp(output, verificationOutput, height*width*sizeof(uchar)) == 0)
    std::cout<<"Pass\n";
  else
    std::cout<<"Fail\n";

  /* release program resources (input memory etc.) */
  if(input) free(input);
  if(key) free(key);
  if(expandedKey) free(expandedKey);
  if(roundKey) free(roundKey);
  if(output) free(output);
  if(verificationOutput) free(verificationOutput);
  return 0;
}

