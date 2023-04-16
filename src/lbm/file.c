#include <math.h>
#include <stdio.h>
#include "config.h"
#include "lbm_1d_array.h"

void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
  const int litteBigEndianTest = 1;
  if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
    const char* vPtr = (char*) v;
    char buffer[sizeof( OUTPUT_PRECISION )];
    int i;
    for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
      buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];
    if (fwrite(buffer, sizeof(OUTPUT_PRECISION), 1, file) != 1)
      printf("WARNING: reading file error\n");
  } else {/* little endian */
    if (fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file) != 1)
      printf("WARNING: reading file error\n");
  }
}

void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
  const int litteBigEndianTest = 1;
  if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {/* big endian */
    char* vPtr = (char*) v;
    char buffer[sizeof( OUTPUT_PRECISION )];
    int i;
    if (fread(buffer, sizeof(OUTPUT_PRECISION), 1, file) != 1)
      printf("WARNING: reading file error\n");
    for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
      vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
  } else { /* little endian */
    if (fread(v, sizeof(OUTPUT_PRECISION), 1, file) != 1)
      printf("WARNING: reading file error\n");
  }
}

void LBM_compareVelocityField( LBM_Grid grid, const char* filename, const int binary ) {
  int x, y, z;
  float rho, ux, uy, uz;
  OUTPUT_PRECISION fileUx, fileUy, fileUz, dUx, dUy, dUz, diff2, maxDiff2 = -1e+30;
  FILE* file = fopen( filename, (binary ? "rb" : "r") );
  for ( z = 0; z < SIZE_Z; z++ ) {
    for ( y = 0; y < SIZE_Y; y++ ) {
      for ( x = 0; x < SIZE_X; x++ ) {
        rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
              + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
              + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
              + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
              + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
              + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
              + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
              + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
              + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
              + GRID_ENTRY( grid, x, y, z, WB );
        ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
             + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
             + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
             + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
             - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
        uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
             + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
             - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
             + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
             - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
        uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
             + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
             + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
             + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
             + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
        ux /= rho;
        uy /= rho;
        uz /= rho;
        if( binary ) {
          loadValue( file, &fileUx );
          loadValue( file, &fileUy );
          loadValue( file, &fileUz );
        }
        else {
          //fscanf(file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz);
          if (fscanf(file, "%f %f %f\n", &fileUx, &fileUy, &fileUz) < 0)
            printf("WARNING: reading file error\n");
        }
        dUx = ux - fileUx;
        dUy = uy - fileUy;
        dUz = uz - fileUz;
        diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
        if ( diff2 > maxDiff2 ) maxDiff2 = diff2;
      }
    }
  }
  printf("compareVelocityField: maxDiff = %e  ==>  %s\n\n", sqrt( maxDiff2 ),
         sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK");
  fclose( file );
}

void LBM_storeVelocityField( LBM_Grid grid, const char* filename, const int binary ) {
  int x, y, z;
  OUTPUT_PRECISION rho, ux, uy, uz;
  FILE* file = fopen( filename, (binary ? "wb" : "w") );
  for ( z = 0; z < SIZE_Z; z++ ) {
    for ( y = 0; y < SIZE_Y; y++ ) {
      for ( x = 0; x < SIZE_X; x++ ) {
        rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
              + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
              + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
              + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
              + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
              + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
              + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
              + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
              + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
              + GRID_ENTRY( grid, x, y, z, WB );
        ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
             + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
             + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
             + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
             - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
        uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
             + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
             - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
             + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
             - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
        uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
             + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
             + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
             + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
             + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
        ux /= rho;
        uy /= rho;
        uz /= rho;
        if( binary ) {
          storeValue( file, &ux );
          storeValue( file, &uy );
          storeValue( file, &uz );
        } else fprintf( file, "%e %e %e\n", ux, uy, uz );
      }
    }
  }
  fclose( file );
}

void LBM_showGridStatistics( LBM_Grid grid ) {
  int nObstacleCells = 0,
      nAccelCells    = 0,
      nFluidCells    = 0;
  float ux, uy, uz;
  float minU2  = 1e+30, maxU2  = -1e+30, u2;
  float minRho = 1e+30, maxRho = -1e+30, rho;
  float mass = 0;
  SWEEP_VAR
  SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
  rho = + LOCAL( grid, C  ) + LOCAL( grid, N  )
        + LOCAL( grid, S  ) + LOCAL( grid, E  )
        + LOCAL( grid, W  ) + LOCAL( grid, T  )
        + LOCAL( grid, B  ) + LOCAL( grid, NE )
        + LOCAL( grid, NW ) + LOCAL( grid, SE )
        + LOCAL( grid, SW ) + LOCAL( grid, NT )
        + LOCAL( grid, NB ) + LOCAL( grid, ST )
        + LOCAL( grid, SB ) + LOCAL( grid, ET )
        + LOCAL( grid, EB ) + LOCAL( grid, WT )
        + LOCAL( grid, WB );
  if ( rho < minRho ) minRho = rho;
  if ( rho > maxRho ) maxRho = rho;
  mass += rho;
  if ( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
    nObstacleCells++;
  }
  else {
    if ( TEST_FLAG_SWEEP( grid, ACCEL ))
      nAccelCells++;
    else
      nFluidCells++;
    ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
         + LOCAL( grid, NE ) - LOCAL( grid, NW )
         + LOCAL( grid, SE ) - LOCAL( grid, SW )
         + LOCAL( grid, ET ) + LOCAL( grid, EB )
         - LOCAL( grid, WT ) - LOCAL( grid, WB );
    uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
         + LOCAL( grid, NE ) + LOCAL( grid, NW )
         - LOCAL( grid, SE ) - LOCAL( grid, SW )
         + LOCAL( grid, NT ) + LOCAL( grid, NB )
         - LOCAL( grid, ST ) - LOCAL( grid, SB );
    uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
         + LOCAL( grid, NT ) - LOCAL( grid, NB )
         + LOCAL( grid, ST ) - LOCAL( grid, SB )
         + LOCAL( grid, ET ) - LOCAL( grid, EB )
         + LOCAL( grid, WT ) - LOCAL( grid, WB );
    u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
    if ( u2 < minU2 ) minU2 = u2;
    if ( u2 > maxU2 ) maxU2 = u2;
  }
  SWEEP_END
  printf( "LBM_showGridStatistics:\n"
      "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
      "\tminRho: %8.4f maxRho: %8.4f mass: %e\n"
      "\tminU: %e maxU: %e\n\n",
      nObstacleCells, nAccelCells, nFluidCells,
      minRho, maxRho, mass,
      sqrt( minU2 ), sqrt( maxU2 ) );
}

