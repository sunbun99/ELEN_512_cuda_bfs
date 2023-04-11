
#ifndef CUDACC
#define CUDACC
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <stdbool.h>
#include <ctype.h>
#include <list>

#include <windows.h>  

#define NUM_NODES 15606
#define NUM_EDGES 45878

typedef struct
{
	int start;     // Index of first adjacent node in Ea	
	int length;    // Number of adjacent nodes 
} Node;

__global__ void CUDA_BFS_KERNEL(Node* Va, int* Ea, bool* Fa, bool* Xa, int* Ca, bool* done)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > NUM_NODES)
		*done = false;


	if (Fa[id] == true && Xa[id] == false)
	{
		printf("%d ", id); //This printf gives the order of vertices in BFS	
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads();
		// k = 0;
		//int i;
		int start = Va[id].start;
		int end = start + Va[id].length;
		for (int i = start; i < end; i++)
		{
			int nid = Ea[i];

			if (Xa[nid] == false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}

		}

	}

}

// The BFS frontier corresponds to all the nodes being processed at the current level.


int main()
{
	FILE* myFile;
	myFile = fopen("4elt.graph.txt", "r");
	// myFile = fopen("mdual.graph", "r");
	// myFile = fopen("citationCiteseer.graph", "r");

	//myFile = fopen("test.graph", "r");
	
	if (myFile == NULL)
	{
		printf("Can't open the file\n");
		return 1;
	}

	//Node node[NUM_NODES];
	//int edges[NUM_EDGES];

	std::list<Node> nodes;

	Node n = { 0,0 };
	nodes.push_back(n); // add dummy element to help initialize list

	std::list<int> edges_;

	char buffer[10000];
	char* pbuff;
	char* tmpBuff;
	int value;
	int lineNum = 0;
	int edgeIndex = 0;
	int graph_stat = 0;
	int is_num = 0;
	int currLineNum = 0;
	int currLen = 0;
	int length = 0;

	while (1)
	{
		if (fgets(buffer, sizeof buffer, myFile) == 0)
			break;
		// printf("Line contains");
		pbuff = buffer;

		if (lineNum != 0) {
			//node[lineNum - 1].start = edgeIndex;
			//node[lineNum - 1].length = 0;
			currLineNum = lineNum - 1;
			currLen = 0;
		}

		while (1)
		{
			if (*pbuff == '\n')
				break;
			if (*pbuff == '\0')
				break;
			tmpBuff = pbuff;
			is_num = 0;
			while (*tmpBuff != '\0') {
				if (!isspace((unsigned char)*tmpBuff))
					is_num = 1;
				tmpBuff++;
			}
			if (is_num == 0)
				break;
			value = strtol(pbuff, &pbuff, 10);
			if (graph_stat == 0 && lineNum == 0)
			{
				printf("Creating graph of size %d \n", value);
				if (NUM_NODES != value)
				{
					printf("Node count not configured properly for loaded graph\n");
					return 1;
				}
				graph_stat = 1;
			}
			else if (lineNum == 0 && graph_stat == 1)
			{
				printf("%s", pbuff);
				printf("Expecting %d edges \n", value);
				if (NUM_EDGES != value)
				{
					printf("Edge count not configured properly for loaded graph\n");
					return 1;
				}
			}
			else
			{
				// printf("%d", value);
				/*
				node[lineNum - 1].length++;
				edges[edgeIndex] = value;*/
				edgeIndex++;
				currLen++;
				edges_.push_back(value);
				length++;
			}
			// if (value == 0)
			//     return 1;
		}
		if (lineNum != 0)
		{
			//Node n;
			n.length = length;
			n.start = nodes.back().start + nodes.back().length;
			nodes.push_back(n);
			length = 0;
		}
		lineNum++;
	}

	nodes.pop_front();  //remove dummy element from front

	Node* node;
	node = (Node*)malloc(nodes.size() * sizeof(Node));
	std::copy(nodes.begin(), nodes.end(), node);
	int* edges;
	edges = (int*)malloc(edges_.size() * sizeof(int));
	std::copy(edges_.begin(), edges_.end(), edges);



	//int edgesSize = 2 * NUM_NODES;
	/*
	int edges[NUM_NODES];

	node[0].start = 0;
	node[0].length = 2;

	node[1].start = 2;
	node[1].length = 1;

	node[2].start = 3;
	node[2].length = 1;

	node[3].start = 4;
	node[3].length = 1;

	node[4].start = 5;
	node[4].length = 0;

	edges[0] = 1;
	edges[1] = 2;
	edges[2] = 4;
	edges[3] = 3;
	edges[4] = 4; */

	//bool frontier[NUM_NODES] = { false };
	//bool visited[NUM_NODES] = { false };
	//int cost[NUM_NODES] = { 0 };

	bool* frontier;
	//frontier = (bool*)malloc(nodes.size() * sizeof(bool));
	frontier = (bool*)calloc(nodes.size(), sizeof(bool));

	bool* visited;
	//visited = (bool*)malloc(nodes.size() * sizeof(bool));
	visited = (bool*)calloc(nodes.size(), sizeof(bool));

	int* cost;
	//cost = (int*)malloc(nodes.size() * sizeof(int));
	cost = (int*)calloc(nodes.size(), sizeof(int));

	int const source = 0;
	frontier[source] = true;

	Node* Va;
	cudaMalloc((void**)&Va, sizeof(Node) * nodes.size());
	cudaMemcpy(Va, node, sizeof(Node) * nodes.size(), cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc((void**)&Ea, sizeof(Node) * nodes.size());
	cudaMemcpy(Ea, edges, sizeof(Node) * nodes.size(), cudaMemcpyHostToDevice);

	bool* Fa;
	cudaMalloc((void**)&Fa, sizeof(bool) * nodes.size());
	cudaMemcpy(Fa, frontier, sizeof(bool) * nodes.size(), cudaMemcpyHostToDevice);

	bool* Xa;
	cudaMalloc((void**)&Xa, sizeof(bool) * nodes.size());
	cudaMemcpy(Xa, visited, sizeof(bool) * nodes.size(), cudaMemcpyHostToDevice);

	int* Ca;
	cudaMalloc((void**)&Ca, sizeof(int) * nodes.size());
	cudaMemcpy(Ca, cost, sizeof(int) * nodes.size(), cudaMemcpyHostToDevice);



	//int num_blks = (512 / NUM_NODES)+1;
	int num_blks = NUM_NODES / 512;
	int threads = 512;



	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	printf("\n\n");
	int count = 0;

	printf("Order: \n\n");
	do {
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_BFS_KERNEL <<< num_blks, threads >>> (Va, Ea, Fa, Xa, Ca, d_done);
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!done);




	cudaMemcpy(cost, Ca, sizeof(int) * nodes.size(), cudaMemcpyDeviceToHost);

	printf("Number of times the kernel is called : %d \n", count);


	printf("\nCost: ");
	for (int i = 0; i < NUM_NODES; i++)
		printf("%d    ", cost[i]);
	printf("\n");
	_getch();

}