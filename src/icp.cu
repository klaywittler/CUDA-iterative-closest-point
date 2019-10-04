#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "icp.h"
#include <cublas_v2.h>
#include <cusolverDn.h>
//#include <Eigen/Dense>



#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define index(i,j,ld) (((j)*(ld))+(i))

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 25.0f

/*****************************
* Self defined configuration *
******************************/
/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
int startSize;
int targetSize;

dim3 threadsPerBlock(blockSize);

glm::vec3 *dev_pos;
glm::vec3 *dev_start;
glm::vec3 *dev_target;
glm::vec3 *dev_color;
int *dev_cor;

glm::vec3 *host_start;
glm::vec3 *host_target;
int *cor;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

__global__ void kernColorBuffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}
/**
* Initialize memory, update some globals
*/
void ICP::initSimulation(std::vector<glm::vec3> start, std::vector<glm::vec3> target) {
	startSize = start.size();
	targetSize = target.size();
	numObjects = startSize + targetSize;
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// Don't forget to cudaFree in  ICP::endSimulation.
	cudaMalloc((void**)&dev_start, startSize * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_start failed!");

	cudaMalloc((void**)&dev_target, targetSize * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_target failed!");

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_color, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

	cudaMalloc((void**)&dev_cor, startSize * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	// move start and target points to GPU
	cudaMemcpy(dev_start, &start[0], startSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_target, &target[0], targetSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_pos, &start[0], startSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_pos[startSize], &target[0], targetSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	//set colors for points
	dim3 startBlocks((numObjects + blockSize - 1) / blockSize);
	dim3 targetBlocks((numObjects + blockSize - 1) / blockSize);
	kernColorBuffer << <startBlocks, blockSize >> > (startSize, dev_color, glm::vec3(0, 1, 0));
	kernColorBuffer << <targetBlocks, blockSize >> > (targetSize, &dev_color[startSize], glm::vec3(1, 0, 0));

	cudaDeviceSynchronize();

	host_start = (glm::vec3*)malloc(startSize * sizeof(glm::vec3));
	host_target = (glm::vec3*)malloc(targetSize * sizeof(glm::vec3));
	cor = (int*)malloc(numObjects * sizeof(int));
}


void ICP::endSimulation() {
	cudaFree(dev_pos);
	cudaFree(dev_start);
	cudaFree(dev_target);
	cudaFree(dev_cor);
	cudaFree(dev_color);

	free(host_start);
	free(host_target);
	free(cor);

}


/******************
* copyICPToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyColorToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyICPToVBO CUDA kernel.
*/
void ICP::copyToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyColorToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_color, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

typedef struct _matrixSize {
	int WA, HA, WB, HB, WC, HC;
} sMatrixSize;

void matrixMultiply(cublasHandle_t* handle, sMatrixSize &matrix_size, float *d_A, float *d_B, float *d_C) {
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.WA, matrix_size.WB, matrix_size.HA, &alpha, d_A, matrix_size.HA, d_B, matrix_size.HB, &beta, d_C, matrix_size.HC);
	checkCUDAError("matrix multiply");
}

void printMat(float*P, int uWP, int uHP) {
	int i, j;
	for (i = 0; i < uHP; i++) {
		for (j = 0; j < uWP; j++)
			printf(" %f ", P[index(i, j, uHP)]);
		printf("\n");
	}
}

void correspondenceCPU() {
	for (int i = 0; i < startSize; i++) {
		float best = glm::distance(host_start[i], host_target[0]);
		cor[i] = 0; //startSize
		for (int j = 1; j < targetSize; j++) {
			float dist = glm::distance(host_start[i], host_target[j]);
			if (dist < best) {
				cor[i] = j; // j + startSize
				best = dist;
			}

		}
	}
}

void procrustesCPU() {

}

void ICP::stepCPU() {
	glm::vec3 mu_start(0.0f);
	glm::vec3 mu_target(0.0f);

	correspondenceCPU();
	procrustesCPU();

}


__global__ void correspondenceGPU(int startSize, int targetSize, glm::vec3 *dev_start, glm::vec3 *dev_target, int *cor){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= startSize) {
		return;
	}
	float best = glm::distance(dev_start[index], dev_target[0]);
	cor[index] = 0;
	for (int j = 1; j < targetSize; j++) {
		float dist = glm::distance(dev_start[index], dev_target[j]);
		if (dist < best) {
			cor[index] = j;
			best = dist;
		}

	}
}


__global__ void shuffleTarget(int startSize, glm::vec3 *dev_target, glm::vec3 *cor_target, int *cor) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= startSize) {
		return;
	}
	cor_target[index] = dev_target[cor[index]];
}


__global__ void translate(int n, glm::vec3 *pos, glm::vec3 T) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	pos[index] = pos[index] + T;
}


void ICP::stepGPU() {
	dim3 startblocksPerGrid((startSize + blockSize - 1) / blockSize);
	dim3 targetblocksPerGrid((targetSize + blockSize - 1) / blockSize);
	// mean center both data sets
	thrust::device_ptr<glm::vec3> thrust_target(dev_target);
	thrust::device_ptr<glm::vec3> thrust_start(dev_start);

	glm::vec3 targetMu = glm::vec3(thrust::reduce(thrust_target, thrust_target + targetSize)) / float(targetSize);
	glm::vec3 startMu = glm::vec3(thrust::reduce(thrust_start, thrust_start + startSize)) / float(startSize);

	translate << <targetblocksPerGrid, blockSize >> > (targetSize, dev_target, -targetMu);
	translate << <startblocksPerGrid, blockSize >> > (startSize, dev_start, -startMu);

	// find correspondences
	glm::vec3 *cor_target;
	cudaMalloc((void**)&cor_target, startSize * sizeof(glm::vec3));

	correspondenceGPU <<<startblocksPerGrid, blockSize >>> (startSize, targetSize, dev_start, dev_target, dev_cor);
	shuffleTarget <<<startblocksPerGrid, blockSize >>> (startSize, dev_target, cor_target, cor);

	// multiply cor_target and dev_start for svd
	//sMatrixSize matrix_size = { WA, HA, WB, HB, WB, HA };

	//cublasHandle_t handle;
	//cublasCreate(&handle);

	//matrixMultiply(&handle, matrix_size, dev_start, cor_target, d_C);

	// svd


	// multiply for R, rotation

	// find T, tranlations

	// move start set

	cudaFree(cor_target);
}


void indexInit(float *data, int size) {
	for (int i = 0; i < size; ++i)
		data[i] = (float)i;
}

void matrixTest() {
	int HA = 3, WA = 3, HB = 3, WB = 1;
	sMatrixSize matrix_size = { WA, HA, WB, HB, WB, HA };

	// allocate host memory for matrices A and B
	unsigned int size_A = matrix_size.WA * matrix_size.HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = matrix_size.WB * matrix_size.HB;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// initialize host memory
	indexInit(h_A, size_A);
	indexInit(h_B, size_B);

	// allocate device memory
	float *d_A, *d_B, *d_C;
	unsigned int size_C = matrix_size.WC * matrix_size.HC;
	unsigned int mem_size_C = sizeof(float) * size_C;

	// allocate host memory for the result
	float *h_C = (float *)malloc(mem_size_C);

	cudaMalloc((void **)&d_A, mem_size_A);
	cudaMalloc((void **)&d_B, mem_size_B);
	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_C, mem_size_C);

	// setup execution parameters
	dim3 threads(blockSize, blockSize);
	dim3 grid(matrix_size.HB / threads.x, matrix_size.WA / threads.y);

	// create and start timer
	printf("Computing result using CUBLAS... \n");

	cublasHandle_t handle;
	cublasCreate(&handle);

	matrixMultiply(&handle, matrix_size, d_A, d_B, d_C);

	// copy result from device to host
	cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	// Destroy the handle
	cublasDestroy(handle);

	printf("\n Matriz A: \n");
	printMat(h_A, matrix_size.WA, matrix_size.HA);
	printf("\n Matriz B: \n ");
	printMat(h_B, matrix_size.WB, matrix_size.HB);
	printf("\n Matriz C: \n");
	printMat(h_C, matrix_size.WC, matrix_size.HC);
	printf("\n\n");

	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void svdTest() {
	int HA = 3, WA = 3, HB = 3, WB = 1;
	sMatrixSize matrix_size = { WA, HA, WB, HB, WB, HA };

	int work_size = 0;
	int *devInfo; cudaMalloc(&devInfo, sizeof(int));

	// allocate host memory for matrices A and B
	unsigned int size_C = matrix_size.WC * matrix_size.HC;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float *h_C = (float *)malloc(mem_size_C);
	float *d_C;

	int Nrows = matrix_size.HC;
	int Ncols = matrix_size.WC;
	// initialize host memory
	indexInit(h_C, size_C);
	cudaMalloc((void **)&d_C, mem_size_C);
	cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

	// --- host side SVD results space
	float *h_U = (float *)malloc(Nrows * Nrows * sizeof(float));
	float *h_V = (float *)malloc(Ncols * Ncols * sizeof(float));
	float *h_S = (float *)malloc(std::min(Nrows, Ncols) * sizeof(float));

	// --- device side SVD workspace and matrices
	float *d_U; cudaMalloc(&d_U, Nrows * Nrows * sizeof(float));
	float *d_V; cudaMalloc(&d_V, Ncols * Ncols * sizeof(float));
	float *d_S; cudaMalloc(&d_S, std::min(Nrows, Ncols) * sizeof(float));


	// --- CUDA solver initialization
	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	// --- CUDA SVD initialization
	cusolverDnSgesvd_bufferSize(solver_handle, matrix_size.HC, matrix_size.WC, &work_size);
	float *work; cudaMalloc(&work, work_size * sizeof(float));

	// --- CUDA SVD execution
	cusolverDnSgesvd(solver_handle, 'A', 'A', matrix_size.HC, matrix_size.WC, d_C, matrix_size.HC, d_S, d_U, matrix_size.HC, d_V, matrix_size.WC, work, work_size, NULL, devInfo);


	std::cout << "Singular values\n";
	for (int i = 0; i < std::min(Nrows, Ncols); i++)
		std::cout << "d_S[" << i << "] = " << h_S[i] << std::endl;

	std::cout << "\nLeft singular vectors - For y = A * x, the columns of U span the space of y\n";
	for (int j = 0; j < Nrows; j++) {
		printf("\n");
		for (int i = 0; i < Nrows; i++)
			printf("U[%i,%i]=%f\n", i, j, h_U[j*Nrows + i]);
	}

	std::cout << "\nRight singular vectors - For y = A * x, the columns of V span the space of x\n";
	for (int i = 0; i < Ncols; i++) {
		printf("\n");
		for (int j = 0; j < Ncols; j++)
			printf("V[%i,%i]=%f\n", i, j, h_V[j*Ncols + i]);
	}

	cusolverDnDestroy(solver_handle);

	cudaFree(d_C);
	cudaFree(d_U);
	cudaFree(d_S);
	cudaFree(d_V);
	cudaFree(devInfo);
	cudaFree(work);
	free(h_C);
	free(h_U);
	free(h_S);
	free(h_V);
}

void ICP::unitTest() {
	//Eigen::MatrixXd m(2, 2);
	//m(0, 0) = 3;
	//m(1, 0) = 2.5;
	//m(0, 1) = -1;
	//m(1, 1) = m(1, 0) + m(0, 1);
	//std::cout << m << std::endl;
	matrixTest();
	svdTest();
	
	return;
}
