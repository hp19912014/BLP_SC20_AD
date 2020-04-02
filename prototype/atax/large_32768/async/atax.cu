/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "./polybenchUtilFuncts.h"
//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#define NX 32768
#define NY 32768

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

/* Pipeline */
#define NUM_CHUNK 8
#define CHUNK_SIZE NX/NUM_CHUNK
#define NUM_STREAMS 8


#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	int i, fail;
	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

__global__ void atax_kernel1_chunk(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp, int chunk)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = CHUNK_SIZE*chunk;

	if ( i < CHUNK_SIZE)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i+offset] += A[(i+offset) * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2_chunk(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp, int chunk)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i= CHUNK_SIZE*chunk; i < CHUNK_SIZE*(chunk+1); i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu)
{

	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX);

	cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaStream_t streams[NUM_STREAMS];
	for (int i=0; i< NUM_STREAMS; i++)
		cudaStreamCreate(&(streams[i]));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)CHUNK_SIZE) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	cudaEventRecord(start);
		
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

	for (int i = 0 ; i < NUM_CHUNK; i++){	
		cudaMemcpyAsync(A_gpu+i*CHUNK_SIZE*NY, A+i*CHUNK_SIZE*NY, sizeof(DATA_TYPE) * NY * CHUNK_SIZE, cudaMemcpyHostToDevice, streams[i % NUM_STREAMS]);
		atax_kernel1_chunk<<< grid1, block,0,streams[i % NUM_STREAMS] >>>(A_gpu,x_gpu,tmp_gpu,i);
		atax_kernel2_chunk<<< grid2, block,0,streams[i % NUM_STREAMS] >>>(A_gpu,y_gpu,tmp_gpu,i);
	}

	cudaDeviceSynchronize();
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);
	

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
}


int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;

	/*
	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));
	*/

	cudaHostAlloc((void **)&A, sizeof(DATA_TYPE) * NX * NY, cudaHostAllocPortable);
	cudaHostAlloc((void **)&x, sizeof(DATA_TYPE) * NY, cudaHostAllocPortable);
	cudaHostAlloc((void **)&y, sizeof(DATA_TYPE) * NY, cudaHostAllocPortable);
	cudaHostAlloc((void **)&y_outputFromGpu, sizeof(DATA_TYPE) * NY, cudaHostAllocPortable);
	cudaHostAlloc((void **)&tmp, sizeof(DATA_TYPE) * NX, cudaHostAllocPortable);


	init_array(x, A);

	GPU_argv_init();
	ataxGpu(A, x, y, tmp, y_outputFromGpu);
	
	
	t_start = rtclock();
	atax_cpu(A, x, y, tmp);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(y, y_outputFromGpu);
	
	cudaFree(A);
	cudaFree(x);
	cudaFree(y);
	cudaFree(y_outputFromGpu);
	cudaFree(tmp);

  	return 0;
}

