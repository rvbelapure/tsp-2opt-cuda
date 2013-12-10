#ifndef __TSP_H
#define __TSP_H

using namespace std;

#define MAX_ARR_LEN	200
#define MAX_CITIES	4096	/* keep this value power of 2 as this is size of shared memory in kernel */

#define PERCENT_ERROR 0.5

typedef struct _city
{
	float x;
	float y;
}city;

int load_cities_data(char *filename, city * city_list);
void generate_tour(vector<unsigned int> &tour, unsigned int size, city * city_list);
unsigned long dist(unsigned int index1, unsigned int index2, city * city_list);
unsigned long get_optimal_solution(char *filename);

void run(vector<unsigned int> &tour, unsigned int size, city * city_list, unsigned long optimal_soln);

#endif
