#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <vector>
#include <string.h>
#include <math.h>

#include "tsp.h"

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Usage : $ %s <city-data-filename>\n", argv[0]);
		exit(1);
	}
	char * filename = argv[1];
	unsigned long optimal = get_optimal_solution(filename);

	city * city_list;
	vector<unsigned int> tour;
	unsigned int cities = 0;

	city_list = (city *) malloc(sizeof(city) * MAX_CITIES);
	if (city_list == NULL) 
	{
		perror("mem allocation");
		exit(2);
	}
	cities = load_cities_data(filename, city_list);
	generate_tour(tour, cities, city_list);
	run(tour, cities, city_list, optimal);
	free(city_list);
}


int load_cities_data(char *filename, city * city_list)
{ 
	int i, ch, count, cities;
	int num;
	float x, y;
	char str[512];

	FILE * f = fopen(filename, "r+t");
	if (f == NULL) 
	{
		perror("input file open");
		exit(3);
	}

	ch = getc(f);  while (ch != '\n') ch = getc(f);			/* Reading NAME */
	ch = getc(f);  while (ch != '\n') ch = getc(f);			/* Reading TYPE */
	ch = getc(f);  while (ch != '\n') ch = getc(f);			/* Reading COMMENT */
	ch = getc(f);  while (ch != ':') ch = getc(f);			/* Reading DIMENSION */

	fscanf(f, "%s\n", str);						/* Get actual value of DIMENSION */
	cities = atoi(str);
	if (cities == 0) 
	{
		fprintf(stderr, "input file format error.\n");
		exit(4);
	}

	ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);	/* Reading EDGE_WEIGHT_TYPE */

	fscanf(f, "%s\n", str);
	if (strcmp(str, "NODE_COORD_SECTION") != 0) {
		fprintf(stderr, "input file format error.\n");
		exit(5);
	}

	count = 0;

	while (fscanf(f, "%d %f %f\n", &num, &x, &y)) {

		city_list[count].x = x;
		city_list[count].y = y;
		count++;
		if (count > cities) 
		{
			fprintf(stderr, "input size mismatch. Truncating remaining cities silently.\n"); 
			break;
		}
		if (count != num)
			fprintf(stderr, "input line mismatch: expected %d instead of %d\n", count, num);

	}

	fscanf(f, "%s", str);					/* Reading EOF */
	fclose(f);

	return count;
}


void generate_tour(vector<unsigned int> &tour, unsigned int size, city * city_list)
{
	vector<unsigned int> temp;

	for (unsigned int i = 1; i < size; i++)
		temp.push_back(i);

	unsigned int id = 0;
	unsigned int minid;
	unsigned long mincost;

	tour.push_back(0);

	while (temp.size() > 0) 
	{
		minid = 0;
		mincost = dist(tour[id], temp[minid], city_list);

		for (unsigned int i = 0; i < temp.size(); i++) 
		{
			if (dist(tour[id], temp[i], city_list) < mincost)	
			{
				mincost = dist(tour[id], temp[i], city_list);
				minid = i;
			}
		}
		tour.push_back(temp[minid]);
		temp.erase(temp.begin()+minid, temp.begin()+minid+1);
		id++;
	}
}

unsigned long dist(unsigned int index1, unsigned int index2, city * city_list) 
{
	float dx, dy;
 	dx = city_list[index1].x - city_list[index2].x;
	dy = city_list[index1].y - city_list[index2].y;
	return (unsigned long)(sqrtf(dx * dx + dy * dy) + 0.5f);
}

unsigned long get_optimal_solution(char *filename)
{
	if  (strstr (filename,"berlin52.tsp")) return 7542;
	else if (strstr (filename,"ch130.tsp")) return 6110;
	else if (strstr (filename,"pr439.tsp")) return 107217;
	else if (strstr (filename,"kroA100.tsp")) return 21282;
	else if (strstr (filename,"kroE100.tsp")) return 22068;
	else if (strstr (filename,"kroB100.tsp")) return 22141;
	else if (strstr (filename,"kroC100.tsp")) return 20749;
	else if (strstr (filename,"kroD100.tsp")) return 21294;
	else if (strstr (filename,"kroA150.tsp")) return 26524;
	else if (strstr (filename,"kroA200.tsp")) return 29368;
	else if (strstr (filename,"ch150.tsp")) return 6528;
	else if (strstr (filename,"rat195.tsp")) return 2323;
	else if (strstr (filename,"ts225.tsp")) return 126643;
	else if (strstr (filename,"pr226.tsp")) return 80369;
	else if (strstr (filename,"pr264.tsp")) return 49135;
	else if (strstr (filename,"pr299.tsp")) return 48191;
	else if (strstr (filename,"a280.tsp")) return 2579;
	else if (strstr (filename,"att532.tsp")) return 27686;
	else if (strstr (filename,"rat783.tsp")) return 8806;
	else if (strstr (filename,"pr1002.tsp")) return 259045;
	else if (strstr (filename,"vm1084.tsp")) return 239297;
	else if (strstr (filename,"pr2392.tsp")) return 378032;
	else if (strstr (filename,"fl3795.tsp")) return 28772;
	else if (strstr (filename,"pcb3038.tsp")) return 137694;
	else if (strstr (filename,"fnl4461.tsp")) return 182566;
	else if (strstr (filename,"rl5934.tsp")) return 556045;
	else if (strstr (filename,"pla7397.tsp")) return 23260728;
	else if (strstr (filename,"usa13509.tsp")) return 19982859;
	else if (strstr (filename,"d15112.tsp")) return 1573084;
	else if (strstr (filename,"usa15309.tsp")) return 19982859;
	else if (strstr (filename,"d18512.tsp")) return 645238;
	else if (strstr (filename,"sw24978.tsp")) return 855597;
	else if (strstr (filename,"pla33810.tsp")) return 66048945;
	else if (strstr (filename,"pla85900.tsp")) return 142382641;
	else if (strstr (filename,"mona-lisa100K.tsp")) return 5757080;
	else return 0;
}

