#pragma once

typedef unsigned long long AccType;

#define ADJ_SIZE_THREASHOLD 1024
#define FULL_MASK 0xffffffff
#define MAX_PATTERN_SIZE 8
#define MAX_FSM_PATTERN_SIZE 5
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define MAX_THREADS (30 * 1024)
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define MAX_BLOCKS (MAX_THREADS / BLOCK_SIZE)
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

// Vertex Coloring
#define MAX_COLOR 1024

// BFS & SSSP parameters
#define MYINFINITY	1000000000
