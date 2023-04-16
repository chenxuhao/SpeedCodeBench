#ifndef _CONFIG_H_
#define _CONFIG_H_
//#define SIZE   (100)
//#define SIZE_X (1*SIZE)
//#define SIZE_Y (1*SIZE)
//#define SIZE_Z (130)

#define OMEGA (1.95f)

#define OUTPUT_PRECISION float 

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;

#define BOOL int
#define TRUE (-1)
#define FALSE (0)

//Changeable settings
//Padding in each dimension
//Note that the padding in the highest Cartesian dimension 
// must be at least 4 to simplify the kernel by avoiding 
// out-of-bounds access checks.
#define PADDING_X (8)
#define PADDING_Y (0)
#define PADDING_Z (4)

//Pitch in each dimension
#define PADDED_X (SIZE_X+PADDING_X)
#define PADDED_Y (SIZE_Y+PADDING_Y)
#define PADDED_Z (SIZE_Z+PADDING_Z)

#define TOTAL_CELLS (SIZE_X*SIZE_Y*(SIZE_Z))
#define TOTAL_PADDED_CELLS (PADDED_X*PADDED_Y*PADDED_Z)

// Set this value to 1 for GATHER, or 0 for SCATTER
#if 0
#define GATHER
#else
#define SCATTER
#endif

//CUDA block size (not trivially changeable here)
#define BLOCK_SIZE SIZE_X

void storeValue( FILE* file, OUTPUT_PRECISION* v );
void loadValue( FILE* file, OUTPUT_PRECISION* v );
#endif /* _CONFIG_H_ */
