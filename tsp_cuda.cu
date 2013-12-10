#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <set>
#include <algorithm>
#include <queue>

#include "cuda_utils.h"

#include "tsp.h"
#include "tsp_cuda.h"

city * d_clist;
city * h_clist;
unsigned int * device_tour;
unsigned int cities;
__device__ struct best2_out best_2opt;

void run(vector<unsigned int> &tour, unsigned int cities, city * city_list, unsigned long optimal_soln)
{

	CUDA_CHECK_ERROR( cudaMallocHost(&h_clist, sizeof(city) * MAX_CITIES) );
	memcpy(h_clist, city_list, sizeof(city) * cities);
	CUDA_CHECK_ERROR( cudaMalloc((void **)&d_clist, sizeof(city) * cities) );
	CUDA_CHECK_ERROR( cudaMalloc((void **)&device_tour, sizeof(unsigned int) * cities));
	CUDA_CHECK_ERROR( cudaMemcpy(d_clist, h_clist,  sizeof(city) * cities, cudaMemcpyHostToDevice));

	vector <unsigned int> best = tour;
 	unsigned long best_len = tour_len(tour, city_list);
 	unsigned long new_len;
 	 	
	while (optimal_soln < best_len) 
	{
		//2OPT
		local_search_2OPT(tour, best_len);
		new_len = tour_len(tour, city_list);
		if (new_len < best_len) 
		{
			best_len = new_len;
			best = tour;
			fprintf(stdout,"New solution -> Length: %ld = %.5f%% of target\n", 
				best_len, 100.0*(double)best_len/(double)optimal_soln);
		}
		tour = best;
		
		/* 2OPT - reverse */
		for(unsigned int k = 0 ; k < 2 ; k++)
		{
			unsigned int a = rand() % tour.size();
			unsigned int b = rand() % tour.size();
			if (b < a)
			{
				unsigned int utemp = a;
				a = b;
				b = utemp;
			}
			reverse(tour.begin()+a, tour.begin()+b);
		}
	}

	CUDA_CHECK_ERROR( cudaFree(d_clist) );
	CUDA_CHECK_ERROR( cudaFreeHost(h_clist) );
}

unsigned long tour_len(vector<unsigned int> &tour, city * city_list)
{
	unsigned long d = 0;
	for (unsigned int i = 1; i < tour.size(); i++)
		d += dist(tour[i-1], tour[i], city_list);
	d += dist(tour[tour.size()-1], tour[0], city_list);
	return d;	
}

void local_search_2OPT(vector<unsigned int> &tour, unsigned long best_len) 
{
 	struct best2_out zero;
 	struct best2_out out;
 	register int best_i, best_j, best_change;
	int* d_mutex = NULL;
	zero.minchange = 0;
  	zero.i = 0;
  	zero.j = 0;
  	
	CUDA_CHECK_ERROR( cudaMalloc( &d_mutex, sizeof(int)) );
	
	// number of jobs and iterations per thread
	unsigned long long counter = (long)(cities - 2) * (long)(cities - 1 ) / 2;
	unsigned int iter = (counter/(BLOCK_SIZE * GRID_SIZE)) + 1;
	
	best_change = -1;
	while(best_change < 0) 
	{
		cudaMemset( d_mutex, 0, sizeof(int));
		CUDA_CHECK_ERROR( cudaMemcpy(device_tour, &tour[0], sizeof(unsigned int) * cities, cudaMemcpyHostToDevice) );
		CUDA_CHECK_ERROR( cudaMemcpyToSymbol("best_2opt", &zero, sizeof(best2_out), 0, cudaMemcpyHostToDevice) );

		/* XXX : CUDA LAUNCH */
		kernel_localsrch_2opt<<<GRID_SIZE, BLOCK_SIZE>>>(d_clist, tour.size(), device_tour, counter, iter, d_mutex);

		//get ids of edges selected to swap
		CUDA_CHECK_ERROR( cudaMemcpyFromSymbol(&out, "best_2opt", sizeof(struct best2_out), 0, cudaMemcpyDeviceToHost) );

		best_change = out.minchange;
		best_i = out.i;
		best_j = out.j;

		//perform the swap
		if (best_j > best_i) 
		{
			int itemp = best_i;
			best_i = best_j;
			best_j = itemp;
		}

		if (best_change < 0)
			reverse(tour.begin() + best_j, tour.begin() + best_i);

	};
	
	CUDA_CHECK_ERROR( cudaFree(d_mutex) );
}

__global__ void kernel_localsrch_2opt(city * c, int _cities, unsigned int * t, unsigned long counter, unsigned int iterations, int* mutex)
{
	register int local_id = threadIdx.x + blockIdx.x * blockDim.x;
	register int id;
	register unsigned int i, j;
	register unsigned long max = counter;
	// 2-opt move index
	register int change;
	register int packSize = blockDim.x*gridDim.x;
	struct best2_out best;
	register int iter = iterations;
	best.minchange = 999999;

	__shared__ city clist[MAX_CITIES];
	__shared__ int cities;
	__shared__ best2_out best_values[1024];

	cities = _cities;
	for (register int k = threadIdx.x; k < cities; k += blockDim.x)
		clist[k] = c[t[k]];

	__syncthreads();

	#pragma unroll
	for (register int no = 0; no < iter; no++) 
	{	
		id = local_id  + no * packSize;
		if (id < max) 
		{ 	 
			i = (unsigned int)(3+__fsqrt_rn(8.0f*(float)id+1.0f))/2;
			j = id - (i-2)*(i-1)/2 + 1;
			change = d_dist(j, i,clist) + d_dist(i-1, j-1, clist) - d_dist(i-1, i, clist) - d_dist(j-1, j, clist);
			if (change < best.minchange) 
			{
				best.minchange = change;
				best.i = i;
				best.j = j;
				best_values[threadIdx.x] = best;
			}
		}
	}
	__syncthreads();

	//1024 threads
	if ((threadIdx.x < 512) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+512].minchange) )
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+512].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+512].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+512].j;
	}
	__syncthreads();
	
	//512 threads
	if ((threadIdx.x < 256) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+256].minchange)) 
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+256].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+256].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+256].j;
	}
	__syncthreads();

	//256 threads
	if ((threadIdx.x < 128) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+128].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+128].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+128].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+128].j;
	}
	__syncthreads();

	//128 threads
	if ((threadIdx.x < 64) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+64].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+64].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+64].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+64].j;
	}
	__syncthreads();

	//64 threads
	if ((threadIdx.x < 32) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+32].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+32].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+32].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+32].j;
	}
	__syncthreads();

	//32 threads
	if ((threadIdx.x < 16) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+16].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+16].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+16].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+16].j;
	}
	__syncthreads();

	//16 threads
	if ((threadIdx.x < 8) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+8].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+8].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+8].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+8].j;
	}
	__syncthreads();

	//8 threads
	if ((threadIdx.x < 4) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+4].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+4].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+4].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+4].j;
	}
	__syncthreads();

	//4 threads
	if ((threadIdx.x < 2) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+2].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+2].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+2].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+2].j;
	}
	__syncthreads();

	//2 threads
	if ((threadIdx.x < 1) && (best_values[threadIdx.x].minchange > best_values[threadIdx.x+1].minchange))
	{
		best_values[threadIdx.x].minchange = best_values[threadIdx.x+1].minchange;
		best_values[threadIdx.x].i = best_values[threadIdx.x+1].i;
		best_values[threadIdx.x].j = best_values[threadIdx.x+1].j;
	}

	__syncthreads();

	if(threadIdx.x == 0 )
	{

		if (best_values[threadIdx.x].minchange < best_2opt.minchange)
		{
			//inter-block reduction
			atomicMin(&(best_2opt.minchange), best_values[threadIdx.x].minchange);
			while( atomicCAS( mutex, 0, 1 ) != 0 );	/* acquire lock */
			if (best_values[threadIdx.x].minchange == best_2opt.minchange)
				memcpy((void*)&best_2opt, (void*)&best_values[threadIdx.x], sizeof(struct best2_out));
			atomicExch( mutex, 0 );			/* release lock */
		}			
	}						   	
}

__device__ int d_dist(unsigned int i, unsigned int j, city * city_list)
{
	register float dx, dy;	
	dx = city_list[i].x - city_list[j].x;
	dy = city_list[i].y - city_list[j].y;
	return (int)(sqrtf(dx * dx + dy * dy) + 0.5f);
}
