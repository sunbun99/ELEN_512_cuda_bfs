
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#define NUM_NODES 5

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
		int k = 0;
		int i;
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
__global__ void kernel_cuda_simple(
    int *v_adj_list,
    int *v_adj_begin,
    int *v_adj_length,
    int num_vertices,
    int *result,
    bool *still_running)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int v = 0; v < num_vertices; v += num_threads)
    {
        int vertex = v + tid;

        if (vertex < num_vertices)
        {
            for (int n = 0; n < v_adj_length[vertex]; n++)
            {
                int neighbor = v_adj_list[v_adj_begin[vertex] + n];

                if (result[neighbor] > result[vertex] + 1)
                {
                    result[neighbor] = result[vertex] + 1;
                    *still_running = true;
                }
            }
        }
    }
}

int bfs_cuda_simple(
    int *v_adj_list,
    int *v_adj_begin, 
    int *v_adj_length, 
    int num_vertices, 
    int num_edges,
    int start_vertex, 
    int *result)
{
    int *k_v_adj_list;
    int *k_v_adj_begin;
    int *k_v_adj_length;
    int *k_result;
    bool *k_still_running;

    int kernel_runs = 0;
    bool *still_running = new bool[1];
    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;
    bool false_value = false;

    cudaMalloc(&k_v_adj_list, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_begin, sizeof(int) * num_vertices);
    cudaMalloc(&k_v_adj_length, sizeof(int) * num_vertices);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_still_running, sizeof(bool) * 1);

    cudaMemcpy(k_v_adj_list, v_adj_list, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_begin, v_adj_begin, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_length, v_adj_length, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---

    
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    do
    {
        cudaMemcpy(k_still_running, &false_value, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_simple<<<BLOCKS, THREADS>>>(
            k_v_adj_list, 
            k_v_adj_begin, 
            k_v_adj_length, 
            num_vertices, 
            k_result, 
            k_still_running);

        kernel_runs++;

        cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*still_running);

    cudaThreadSynchronize();

    gettimeofday(&t2, NULL);
    long long time = get_elapsed_time(&t1, &t2);

    if (report_time)
    {
        printf("%s,%i,%i,%i,%i,%lld\n", __FILE__, num_vertices, num_edges, BLOCKS, THREADS, time); 
    }


    // --- END MEASURE TIME ---


    cudaMemcpy(result, k_result, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    cudaFree(k_v_adj_list);
    cudaFree(k_v_adj_begin);
    cudaFree(k_v_adj_length);
    cudaFree(k_result);
    cudaFree(k_still_running);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}
__global__ void kernel_cuda_per_edge_basic(int *v_adj_from, int *v_adj_to, int num_edges, int *result, bool *still_running){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = blockDim.x * gridDim.x;
	
	for (int e = 0; e < num_edges; e += num_threads){
		int edge = e + tid;
		if (edge < num_edges){
			int to_vertex = v_adj_to[edge];
			int new_len = result[v_adj_from[edge]] + 1;

			if (new_len < result[to_vertex]){
			result[to_vertex] = new_len;
			*still_running = true;
			}
		}
	}
}

int bfs_cuda_per_edge_basic(
    int *v_adj_list,
    int *v_adj_begin, 
    int *v_adj_length, 
    int num_vertices, 
    int num_edges,
    int start_vertex, 
    int *result)
{
    // Convert data
    // TODO: Check if it is better to allocate only one array
    int *v_adj_from = new int[num_edges];
    int *v_adj_to = new int[num_edges];

    int next_index = 0;
    for (int i = 0; i < num_vertices; i++)
    {
        for (int j = v_adj_begin[i]; j < v_adj_length[i] + v_adj_begin[i]; j++)
        {
            v_adj_from[next_index] = i;
            v_adj_to[next_index++] = v_adj_list[j];
        }
    }

    int *k_v_adj_from;
    int *k_v_adj_to;
    int *k_result;
    bool *k_still_running;

    int kernel_runs = 0;

    fill_n(result, num_vertices, MAX_DIST);
    result[start_vertex] = 0;

    bool *still_running = new bool[1];

    cudaMalloc(&k_v_adj_from, sizeof(int) * num_edges);
    cudaMalloc(&k_v_adj_to, sizeof(int) * num_edges);
    cudaMalloc(&k_result, sizeof(int) * num_vertices);
    cudaMalloc(&k_still_running, sizeof(bool) * 1);

    cudaMemcpy(k_v_adj_from, v_adj_from, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_v_adj_to, v_adj_to, sizeof(int) * num_edges, cudaMemcpyHostToDevice);
    cudaMemcpy(k_result, result, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);


    // --- START MEASURE TIME ---


    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    do
    {
        *still_running = false;
        cudaMemcpy(k_still_running, still_running, sizeof(bool) * 1, cudaMemcpyHostToDevice);

        kernel_cuda_per_edge_basic<<<BLOCKS, THREADS>>>(
            k_v_adj_from, 
            k_v_adj_to, 
            num_edges, 
            k_result, 
            k_still_running);

        kernel_runs++;

        cudaMemcpy(still_running, k_still_running, sizeof(bool) * 1, cudaMemcpyDeviceToHost);
    } while (*still_running);

    cudaThreadSynchronize();

    gettimeofday(&t2, NULL);
    long long time = get_elapsed_time(&t1, &t2);

    if (report_time)
    {
        printf("%s,%i,%i,%i,%i,%lld\n", __FILE__, num_vertices, num_edges, BLOCKS, THREADS, time); 
    }


    // --- END MEASURE TIME ---


    cudaMemcpy(result, k_result, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);

    cudaFree(k_v_adj_from);
    cudaFree(k_v_adj_to);
    cudaFree(k_result);
    cudaFree(k_still_running);

    free(v_adj_from);
    free(v_adj_to);

    // printf("%i kernel runs\n", kernel_runs);

    return time;
}

// The BFS frontier corresponds to all the nodes being processed at the current level.

//instead of creating a graph in the way that we do in the method seen below we should do CSR??
int main()
{




	Node node[NUM_NODES];


	//int edgesSize = 2 * NUM_NODES;
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
	edges[4] = 4;

	bool frontier[NUM_NODES] = { false };
	bool visited[NUM_NODES] = { false };
	int cost[NUM_NODES] = { 0 };

	int source = 0;
	frontier[source] = true;

	Node* Va;
	cudaMalloc((void**)&Va, sizeof(Node) * NUM_NODES);
	cudaMemcpy(Va, node, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

	int* Ea;
	cudaMalloc((void**)&Ea, sizeof(Node) * NUM_NODES);
	cudaMemcpy(Ea, edges, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

	bool* Fa;
	cudaMalloc((void**)&Fa, sizeof(bool) * NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

	bool* Xa;
	cudaMalloc((void**)&Xa, sizeof(bool) * NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

	int* Ca;
	cudaMalloc((void**)&Ca, sizeof(int) * NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);



	int num_blks = 1;
	int threads = 5;



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
		CUDA_BFS_KERNEL << <num_blks, threads >> > (Va, Ea, Fa, Xa, Ca, d_done);
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!done);




	cudaMemcpy(cost, Ca, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);

	printf("Number of times the kernel is called : %d \n", count);


	printf("\nCost: ");
	for (int i = 0; i < NUM_NODES; i++)
		printf("%d    ", cost[i]);
	printf("\n");
	_getch();

}
