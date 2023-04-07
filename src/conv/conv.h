#ifndef CONV_H
#define CONV_H

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

// Reference row convolution filter
void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);


void convolutionRows(
    float* dst,
    const float* src,
    const float* kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

void convolutionColumns(
    float* dst,
    const float* src,
    const float* kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

#endif
