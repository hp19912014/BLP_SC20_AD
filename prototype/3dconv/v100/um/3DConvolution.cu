/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
#define NI 256
#define NJ 256
#define NK 256

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef double DATA_TYPE;

#define NUM_STREAMS 3
#define NUM_CHUNKS 16
#define CHUNK_SIZE NI/NUM_CHUNKS		//256/2=256
void conv3D(DATA_TYPE* A, DATA_TYPE* B)
{
	int i, j, k;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
			{
				//printf("i:%d\nj:%d\nk:%d\n", i, j, k);
				B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
			}
		}
	}
}


void init(DATA_TYPE* A)
{
	int i, j, k;

	for (i = 0; i < NI; ++i)
    	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
			}
		}
	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu)
{
	int i, j, k, fail;
	fail = 0;
	
	// Compare result from cpu and gpu...
	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK - 1; ++k) // 2
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_outputFromGpu[i*(NK * NJ) + j*NK + k]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( 0 );
}


__global__ void convolution3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;


	if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c21 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c23 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c31 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]  +  c33 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)]
					     +   c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)]  +  c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)]   
					     +   c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)]  +  c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  
					     +   c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]  +  c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  
					     +   c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]  +  c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]  
					     +   c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)];
	}
}


void convolution3DCuda(DATA_TYPE* A, DATA_TYPE* B_outputFromGpu)
{
	//double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	
	 cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
       
	cudaEventRecord(start);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));
	
	int i;
	for (i = 1; i < NI - 1; ++i) // 0
	{
		convolution3D_kernel<<< grid, block >>>(A_gpu, B_gpu, i);
	}

	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyDeviceToHost);
	
	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        fprintf(stdout,"GPU RunTime= %.3f Ms \n",  elapsedTimeInMs);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
}

void convolution3DCuda_async(DATA_TYPE* A, DATA_TYPE* B_outputFromGpu)
{

	//DATA_TYPE *A_gpu;
	//DATA_TYPE *B_gpu;

	//cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	//cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
	
	 cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
       
	cudaEventRecord(start,0);

	cudaStream_t streams[NUM_STREAMS];
	for (int i=0; i< NUM_STREAMS; i++)
		cudaStreamCreate(&(streams[i]));

	//cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NK) / ((float)block.x) )), (size_t)(ceil( ((float)NJ) / ((float)block.y) )));
	
	int i,c;
	//input initialization
	cudaMemPrefetchAsync(A, sizeof(DATA_TYPE)*NJ*NK, 0 ,streams[0]);

	for (c=0; c < NUM_CHUNKS; c++){	
		
		if (c==(NUM_CHUNKS-1)){
		cudaMemPrefetchAsync(A+(c*CHUNK_SIZE+1)*NJ*NK, sizeof(DATA_TYPE)*NJ*NK*(CHUNK_SIZE-1), 0,streams[c % NUM_STREAMS]);
		}else{
		cudaMemPrefetchAsync(A+(c*CHUNK_SIZE+1)*NJ*NK, sizeof(DATA_TYPE)*NJ*NK*CHUNK_SIZE, 0,streams[c % NUM_STREAMS]);
		}
				
		for (i = (c*CHUNK_SIZE ); i < ((c+1)*CHUNK_SIZE); ++i) // 0
		{
		if ((i>=1)&&(i<(NI-1)))
			convolution3D_kernel<<< grid, block,0,streams[c % NUM_STREAMS] >>>(A,B_outputFromGpu, i);
		}

		cudaMemPrefetchAsync(B_outputFromGpu+c*CHUNK_SIZE*NK*NJ,sizeof(DATA_TYPE)*NJ*NK*CHUNK_SIZE,cudaCpuDeviceId,streams[c % NUM_STREAMS]);

	}
	//cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaDeviceSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);


	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
        fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);

	//cudaFree(A_gpu);
	//cudaFree(B_gpu);
}

int main(int argc, char *argv[])
{
	//double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* B_outputFromGpu;

	//A = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	//B = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	//B_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*NK*sizeof(DATA_TYPE));
	cudaMallocManaged((void **)&A, sizeof(DATA_TYPE) * NI * NJ * NK);
        cudaHostAlloc((void **)&B, sizeof(DATA_TYPE) * NI * NJ * NK, cudaHostAllocPortable);
        cudaMallocManaged((void **)&B_outputFromGpu, sizeof(DATA_TYPE) * NI * NJ *NK);

	init(A);
	
	GPU_argv_init();

	convolution3DCuda_async(A, B_outputFromGpu);

	conv3D(A,B);
	compareResults(B, B_outputFromGpu);

	cudaFree(A);
	cudaFree(B);
	cudaFree(B_outputFromGpu);

    	return 0;
}

