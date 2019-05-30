#include "CalculationEngine.hpp"


#include "cuda.h" 
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <stdio.h>


namespace MaybeCuda {



	const int TOTAL_SIZE = 100000;
	const int TOTAL_RUNS = 100;

	__global__ void calcVectorCuda_populate_kernel(int *vec, const int totalSize)
	{

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < totalSize; i += stride) {	
			vec[i] = i + i;
		}
	}

	__global__ void calcVectorCuda_add_kernel(int* results, int *vec, const int totalSize)
	{

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		int result = 0;
		for (int i = index; i < totalSize; i += stride) {
			for (int j = index; j < totalSize; j += stride) {
				result += vec[i];
			}
		}
		// TODO : this is the bug
		//printf("blockIdx.x = %d, ", blockIdx.x);
		//printf("stride = %d, ", stride);
		//printf("threadIdx.x = %d, ", threadIdx.x);
		//printf("blockIdx.x*stride + threadIdx.x = %d\n", blockIdx.x*stride + threadIdx.x);
		results[blockIdx.x*stride + threadIdx.x] = result;
	}

	__global__ void calcVectorCuda_add_final_kernel(int* val, int *results, const int size)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index != 0) { return; }

		int result = 0;
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				result += results[i];
		}
	}
		printf("Result: %d", result);
		val[0] = result;
	}


	int calcVectorNormal() {
		auto start_total_time = std::chrono::high_resolution_clock::now();
		int* vec = (int*)malloc(TOTAL_SIZE * sizeof(int));
		auto start_exec_time = std::chrono::high_resolution_clock::now();
		int ret = 0;
		for (int r = 0; r < TOTAL_RUNS; r++) {
			for (int i = 0; i < TOTAL_SIZE; i++) {
				vec[i] = i + i;
			}
			ret = 0;
			for (int i = 0; i < TOTAL_SIZE; i++) {
				ret += vec[i];
			}
		}

		auto end_exec_time = std::chrono::high_resolution_clock::now();
		free(vec);
		auto end_total_time = std::chrono::high_resolution_clock::now();
		auto time_exec = end_exec_time - start_exec_time;
		auto time_total = end_total_time - start_total_time;
		printf("Execution time on CPU = %dms\n", time_exec / std::chrono::milliseconds(1));
		printf("Total time on CPU = %dms\n", time_exec / std::chrono::milliseconds(1));
		return ret;
	}

	int calcVectorCuda(unsigned int blocks = 1, unsigned int size = 32, bool usingCuda = true) {
		int ret = 0;
		int full_size = blocks * size;
		if (!usingCuda) { return ret; }

		auto start_total_time = std::chrono::high_resolution_clock::now();
		int* vec = NULL;
		if (cudaMalloc(&vec, TOTAL_SIZE * sizeof(int)) != 0) { return ret; }
		int* results = NULL;
		if (cudaMalloc(&results, full_size * sizeof(int)) != 0) { return ret; }
		int* val = NULL;
		if (cudaMallocManaged(&val, 1 * sizeof(int)) != 0) { return ret; }

		auto start_exec_time = std::chrono::high_resolution_clock::now();
		for (int r = 0; r < TOTAL_RUNS; r++) {
			calcVectorCuda_populate_kernel << <blocks, size >> >(vec, TOTAL_SIZE);
			if (cudaDeviceSynchronize() != 0) {
				cudaFree(val);
				cudaFree(results);
				cudaFree(vec);
				return ret;
			}

			val[0] = 0;
			calcVectorCuda_add_kernel << <blocks, size >> > (results, vec, TOTAL_SIZE);
			if (cudaDeviceSynchronize() != 0) {
				cudaFree(val);
				cudaFree(results);
				cudaFree(vec);
				return ret;
			}

			calcVectorCuda_add_final_kernel << <1, 1 >> > (val, results, full_size);
			if (cudaDeviceSynchronize() != 0) {
				cudaFree(val);
				cudaFree(results);
				cudaFree(vec);
				return ret;
			}
		}

		auto end_exec_time = std::chrono::high_resolution_clock::now();
		ret = val[0];
		cudaFree(val);
		cudaFree(results);
		cudaFree(vec);
		auto end_total_time = std::chrono::high_resolution_clock::now();
		auto time_exec = end_exec_time - start_exec_time;
		auto time_total = end_total_time - start_total_time;
		printf("Execution time on GPU = %dms\n", time_exec / std::chrono::milliseconds(1));
		printf("Total time on GPU = %dms\n", time_total / std::chrono::milliseconds(1));
		return ret;
	}

	CalculationEngine::CalculationEngine() {
		this->_usingCuda = cudaSetDevice(0) == 0;
	}
	int CalculationEngine::CalcVector(bool usingCuda)
	{
		if (usingCuda) {
			return calcVectorCuda(1, 32, this->isUsingCuda());
		}
		return calcVectorNormal();
	}
}

