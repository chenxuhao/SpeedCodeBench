#pragma once

typedef unsigned long long AccType;

#define FULL_MASK 0xffffffff
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define BYTESTOMB(memory_cost) ((memory_cost)/(double)(1024 * 1024))

