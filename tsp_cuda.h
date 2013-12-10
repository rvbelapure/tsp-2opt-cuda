#ifndef __TSP_CUDA_H
#define __TSP_CUDA_H

#define BLOCK_SIZE 1024
#define GRID_SIZE  16

struct best2_out
{
	unsigned int i;
	unsigned int j;
	int minchange;
};

unsigned long tour_len(vector<unsigned int> &tour, city * city_list);
void local_search_2OPT(vector<unsigned int> &tour, unsigned long best_len);
__global__ void kernel_localsrch_2opt(city * c, int _cities, unsigned int * t, unsigned long counter, unsigned int iterations, int* mutex);
__device__ int d_dist(unsigned int i, unsigned int j, city * city_list);

#endif
