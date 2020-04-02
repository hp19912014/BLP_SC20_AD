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
#define NX 16384
#define NY 16384

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 1024
#define DIM_THREAD_BLOCK_Y 1

/* spin */
#define DIM_ROW (NX/DIM_THREAD_BLOCK_X)       //16384/1024=16
#define DIM_COLUMN (NY/DIM_THREAD_BLOCK_Y)    //16384/1=16384
#define DIM_BLOCK DIM_ROW*DIM_COLUMN		  //16*16384=262144
#define DIM_BLOCK_VECTOR DIM_BLOCK/4		//65536

#define NUM_SM 32
//#define NUM_SM_HtoD 4
//#define OFFSET NUM_SM_HtoD * DIM_THREAD_BLOCK_X
#define OFFSET DIM_THREAD_BLOCK_X
#define NUM_SM_COMPUTE_tmp 16
#define NUM_SM_COMPUTE_y   16

#define IN_CHUNK_SIZE 32
#define IN_CHUNK NX/IN_CHUNK_SIZE 			//512 
#define IN_CHUNK_OFFSET OFFSET*4*IN_CHUNK_SIZE  		// 4096 * 32 = 131072    


#define NUM_COMPUTE_BLOCK NX/DIM_THREAD_BLOCK_X     //16384/1024=16
#define COPY_CHUNK_NUM IN_CHUNK/NUM_COMPUTE_BLOCK  	// 512/16=32
#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

__device__ int flag_global_read(volatile int * flag, int rid)
{
	return(flag[rid]);
}

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


__global__ void atax_kernel(DATA_TYPE *A, DATA_TYPE *A_host,  DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp, int *flag_A, int* flag_tmp, int* devtaskid)
{
     
	if (blockIdx.x < NUM_SM_COMPUTE_tmp ){  //copy kernel HtoD
		const int idx = threadIdx.x;
                const int bidx = blockIdx.x;
		__shared__ int taskid;
		
		

		// get the first task id
                 if ( idx == 0 ){
                                taskid = atomicAdd(devtaskid,1);
                        }

		__syncthreads();
		//while (taskid < NUM_COMPUTE_BLOCK){
			
			// copy
		/*	
		for (int i = taskid*COPY_CHUNK_NUM ; i< (taskid+1)*COPY_CHUNK_NUM ;i++ ){
                        
			int chunk_offset=i*IN_CHUNK_OFFSET;

                        for (int k = (chunk_offset+idx);k < ( chunk_offset+IN_CHUNK_OFFSET ) ; k+= OFFSET ){
	                        reinterpret_cast<double2*>(A)[k] = reinterpret_cast<double2*>(A_host)[k];
                        }

                  }
		*/
		int chunk_offset = taskid*1024*4096;

                for (int k = (chunk_offset+idx);k < ( chunk_offset+1024*4096 ) ; k+= OFFSET ){
                                reinterpret_cast<double2*>(A)[k] = reinterpret_cast<double2*>(A_host)[k];
                        }

		//__syncthreads();


               // __threadfence();
			
		 //if ( idx <  ){
       		//atomicAdd(&flag_A[taskid*1024+idx],1);
                   //    	}
		__syncthreads();

			//compute

	
			int index = taskid*blockDim.x  + idx;
	                if (index < NX)
        	        {
                	        //while(  flag_global_read(flag_A, index) != 1 );

                        	for(int j=0; j < NY; j++)
                       		{
                        	        tmp[index] += A[index * NY + j] * x[j];
                        	}


                        __threadfence();
                        atomicOr(&flag_tmp[index],1);
                	}

			
			//get next task id
			/*
                        if ( idx == 0 ){
                                taskid = atomicAdd(devtaskid,1);
                        }
			*/
		
		//}
		

	} else if (blockIdx.x < ( NUM_SM_COMPUTE_tmp + NUM_SM_COMPUTE_y)){
	//compute y
	
	int j = (blockIdx.x  - NUM_SM_COMPUTE_tmp) * blockDim.x + threadIdx.x;

        	if (j < NY)
       		 {
                	for(int i=0; i < NX; i++)
                	{
			
			while(  flag_global_read(flag_tmp,i) == 0 );

                       	 y[j] += A[i * NY + j] * tmp[i];
                	}	
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
	
	int *flag_A, *flag_tmp;
	cudaMalloc((void **)&flag_A, sizeof(int) * NX);
	cudaMalloc((void **)&flag_tmp, sizeof(int) * NX);
	
	cudaMemset(flag_A, 0, sizeof(int) * NX);
	cudaMemset(flag_tmp, 0, sizeof(int) * NX);
	
	//cudaMemset(y_gpu, 0, sizeof(DATA_TYPE) * NY);
	//cudaMemset(tmp_gpu, 0, sizeof(DATA_TYPE) * NX);


	cudaEvent_t start,stop;
        float elapsedTimeInMs = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//dynamic task id

	int taskid = 0;
	int *devtaskid;
	cudaMalloc((void**)&devtaskid, sizeof(int));
	cudaMemcpy(devtaskid, &taskid, sizeof(int), cudaMemcpyHostToDevice);


	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(NUM_SM), (size_t)1 );

	void *kernelArgs[] = {
                (void *)&A_gpu,             (void *)&A,
                (void *)&x_gpu,             (void *)&y_gpu,
                (void *)&tmp_gpu,           (void *)&flag_A,
		(void *)&flag_tmp,	    (void *)&devtaskid	
        };	


	cudaEventRecord(start);

        cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
        cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

	cudaLaunchCooperativeKernel((void*)atax_kernel, grid, block, kernelArgs, 0, NULL);
	//atax_kernel<<< grid, block >>>(A_gpu,A,x_gpu,y_gpu, tmp_gpu,flag_A, flag_tmp);
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);
	

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
	fprintf(stdout,"GPU RunTime= %.2f Ms \n",  elapsedTimeInMs);


	// debug copy array A

	/*					
	DATA_TYPE* A_debug;
	cudaHostAlloc((void **)&A_debug, sizeof(DATA_TYPE) * NX * NY, cudaHostAllocPortable);
	cudaMemcpy(A_debug, A_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);	
	compareResults(A,A_debug);
	*/	


	//debug tmp array
	/*	
	DATA_TYPE* tmp_debug;
	cudaHostAlloc((void **)&tmp_debug, sizeof(DATA_TYPE) * NX, cudaHostAllocPortable);
	cudaMemcpy(tmp_debug, tmp_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);
	atax_cpu(A, x, y, tmp);
	compareResults(tmp, tmp_debug);
	*/	
	
	//debug flag_tmp;
	/*
	int* flag_tmp_debug;
        cudaHostAlloc((void **)&flag_tmp_debug, sizeof(int) * NX, cudaHostAllocPortable);
	cudaMemcpy(flag_tmp_debug, flag_tmp, sizeof(int) * NX, cudaMemcpyDeviceToHost);
	for (int i = 0 ; i < NX; i++){
	fprintf(stdout, "%d", flag_tmp_debug[i]);
	}
	*/

	//debug taskid
	/*
	cudaMemcpy(&taskid, devtaskid, sizeof(int), cudaMemcpyDeviceToHost);
	fprintf(stdout, "%d\n", taskid);	
	*/
	//debug flag_A;

	//fprintf(stdout,"%d\n",IN_CHUNK_OFFSET);

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

