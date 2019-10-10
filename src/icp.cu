#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/reduce.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include "utilityCore.hpp"
#include "icp.h"
#include "svd3.h"
#include "kdtree.hpp"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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
#define blockSize 512
/*! Size of the starting area in simulation space. */
#define scene_scale 25.0f
#define ERROR 1e-18f

/*****************************
* Self defined configuration *
******************************/
/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
int startSize;
int targetSize;
bool converged = false;
float prev_error = 0.0;

//dim3 threadsPerBlock(blockSize);
//dim3 startblocksPerGrid(blockSize);
//dim3 targetblocksPerGrid(blockSize);

glm::vec3 *dev_pos;
glm::vec3 *dev_start;
glm::vec3 *dev_target;
glm::vec3 *dev_rgb;
int *dev_cor;
KDtree::KDnode *dev_kd;

glm::vec3 *host_start;
glm::vec3 *host_target;
int *cor;

/******************
* initSimulation *
******************/
__global__ void kernColorBuffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= N) {
		return;
	}
	intBuffer[index] = value;
}

void transformCPU(glm::vec3 *pos, glm::mat3 &R, glm::vec3 &T) {
	for (int i = 0; i < startSize; i++) {
		pos[i] = R * pos[i] + T;
	}
}

__global__ void transform(int n, glm::vec3 *pos, glm::mat3 R, glm::vec3 T) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	pos[index] = R * pos[index] + T;
}


/**
* Initialize memory, update some globals
*/
void ICP::initSimulation(std::vector<glm::vec3> start, std::vector<glm::vec3> target, bool transformScan = true) {
	startSize = start.size();
	targetSize = target.size();
	numObjects = startSize + targetSize;

	dim3 startblocksPerGrid((startSize + blockSize - 1) / blockSize);
	dim3 targetblocksPerGrid((targetSize + blockSize - 1) / blockSize);

	// Don't forget to cudaFree in  ICP::endSimulation.
	cudaMalloc((void**)&dev_start, startSize * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_start failed!");

	cudaMalloc((void**)&dev_target, targetSize * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_target failed!");

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_rgb, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_rgb failed!");

	cudaMalloc((void**)&dev_cor, startSize * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	cudaMalloc((void**)&dev_kd, targetSize * sizeof(KDtree::KDnode));
	checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

	std::unique_ptr< KDtree::KDnode> kd(new KDtree::KDnode[targetSize]);
	KDtree::buildTree(target, kd.get());

	// move start and target points to GPU
	cudaMemcpy(dev_kd, kd.get(), targetSize * sizeof(KDtree::KDnode), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_start, &start[0], startSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_target, &target[0], targetSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	host_start = (glm::vec3*) malloc(startSize * sizeof(glm::vec3));
	host_target = (glm::vec3*) malloc(targetSize * sizeof(glm::vec3));
	cor = (int*) malloc(startSize * sizeof(int));

	memcpy(host_start, &start[0], startSize * sizeof(glm::vec3));
	memcpy(host_target, &target[0], targetSize * sizeof(glm::vec3));

	if (transformScan) {
		//add rotation and translation to start for test;
		glm::vec3 T(5.0, -18.0, 10.0);
		//glm::mat3 R = glm::mat3(glm::vec3(0.058, 0.25, 0.9665), glm::vec3(-0.8995, 0.433, -0.058), glm::vec3(-0.433, -0.866, 0.25)); // does not converge
		glm::mat3 R = glm::mat3(glm::vec3(0.808, 0.25, -0.5335), glm::vec3(0.3995, 0.433, 0.808), glm::vec3(0.433, -0.866, 0.25)); // converges
		// move target set
		transform << <startblocksPerGrid, blockSize >> > (startSize, dev_start, R, T);
		transformCPU(host_start, R, T);
	}

	cudaMemcpy(dev_pos, dev_start, startSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(&dev_pos[startSize], dev_target, targetSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	//set colors for points
	kernColorBuffer << <startblocksPerGrid, blockSize >> > (startSize, dev_rgb, glm::vec3(0, 1, 0));
	kernColorBuffer << <targetblocksPerGrid, blockSize >> > (targetSize, &dev_rgb[startSize], glm::vec3(1, 0, 0));

	cudaDeviceSynchronize();
}

void ICP::endSimulation() {
	cudaFree(dev_pos);
	cudaFree(dev_start);
	cudaFree(dev_target);
	cudaFree(dev_cor);
	cudaFree(dev_rgb);
	cudaFree(dev_kd);

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
  kernCopyColorToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_rgb, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyToVBO failed!");

  cudaDeviceSynchronize();
}

/******************
* stepSimulation *
******************/
void correspondenceCPU(glm::vec3 *start, glm::vec3 *target) {
	for (int i = 0; i < startSize; i++) {
		float best_dist = glm::distance(start[i], target[0]);
		int best_index = 0;
		for (int j = 1; j < targetSize; j++) {
			float dist = glm::distance(start[i], target[j]);
			if (dist < best_dist) {
				best_index = j;
				best_dist = dist;
			}
		}
		cor[i] = best_index;
	}
}

void outerProductCPU(glm::vec3 *target, glm::vec3 *start, glm::mat3 &product) {
	for (int i = 0; i < startSize; i++) {
		product += glm::outerProduct(target[i], start[i]);
	}
}

float convergenceCPU(glm::vec3 *target, glm::vec3 *start) {
	float error = 0;
	for (int i = 0; i < startSize; i++) {
		error += glm::distance(start[i], target[cor[i]]);
	}
	return error/float(startSize);
}

void ICP::stepCPU(bool checkConverge) {
	std::unique_ptr<glm::vec3> temp_start(new glm::vec3[startSize]);
	std::unique_ptr<glm::vec3> temp_target(new glm::vec3[targetSize]);

	memcpy(temp_start.get(), &host_start[0], startSize * sizeof(glm::vec3));
	memcpy(temp_target.get(), &host_target[0], targetSize * sizeof(glm::vec3));

	glm::vec3 startMu(0.0f);
	glm::vec3 targetMu(0.0f);

	// mean center both data sets
	int i = 0;
	while (i < startSize || i < targetSize) {
		if (i < startSize) {
			startMu += host_start[i];
		}
		if (i < targetSize) {
			targetMu += host_target[i];
		}
		i++;
	}
	startMu /= startSize;
	targetMu /= targetSize;
	i = 0;
	while (i < startSize || i < targetSize) {
		if (i < startSize) {
			temp_start.get()[i] -= startMu;
		}
		if (i < targetSize) {
			temp_target.get()[i] -= targetMu;
		}
		i++;
	}
	   
	// find correspondences
	std::unique_ptr<glm::vec3> cor_target(new glm::vec3[startSize]);
	correspondenceCPU(temp_start.get(), temp_start.get());
	// shuffle
	for (int i = 0; i < startSize; i++) {
		cor_target.get()[i] = temp_start.get()[cor[i]];
	}
	
	// outer product of cor_target and dev_start for svd
	glm::mat3 M(0.0f), U, S, V;
	outerProductCPU(cor_target.get(), temp_start.get(), M);
	// svd
	svd(M[0][0], M[1][0], M[2][0], M[0][1], M[1][1], M[2][1], M[0][2], M[1][2], M[2][2],
		U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
		S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
		V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]
	);

	glm::mat3 I(1.0f);
	I[2][2] = glm::determinant(U*glm::transpose(V));
	// multiply for R, rotation
	glm::mat3 R = U * I * glm::transpose(V);
	glm::vec3 T = targetMu - R * startMu;

	// move start set
	transformCPU(host_start, R, T);
	cudaMemcpy(dev_pos, &host_start[0], startSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	if (checkConverge) {
		float error = convergenceCPU(host_target, host_start);
		if (abs(error-prev_error) < ERROR) {
			converged = true;
			std::cout << "CPU average error: " << error << std::endl;
		}
		prev_error = error;
	}

}

__global__ void correspondenceNaive(int startSize, int targetSize, glm::vec3 *dev_start, glm::vec3 *dev_target, int *dev_cor){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= startSize) {
		return;
	}
	float best_dist = glm::distance(dev_start[index], dev_target[0]);
	int best_index = 0;
	for (int j = 1; j < targetSize; j++) {
		float dist = glm::distance(dev_start[index], dev_target[j]);
		if (dist < best_dist) {
			best_dist = dist;
			best_index = j;
		}
	}
	dev_cor[index] = best_index;
}

__global__ void correspondenceKDtree(int startSize, int targetSize, glm::vec3 *dev_start, glm::vec3 *dev_target, int *dev_cor, const KDtree::KDnode *tree) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= startSize) {
		return;
	}
	int best_index = 0, root = 0;
	float best_dist = glm::distance(dev_start[index], dev_target[tree[root].current]);
	bool finished = false, explored = false;

	while (!finished) {
		while (root >= 0) { //while not a leaf node
			const KDtree::KDnode target = tree[root];
			float dist = glm::distance(dev_start[index], dev_target[target.current]);
			if (dist < best_dist) {
				best_index = root;
				best_dist = dist;
				explored = false;
			}
			root = dev_start[index][target.split_axis] < dev_target[target.current][target.split_axis] ? target.left : target.right;
		}
		if (explored) {
			finished = true;
		}
		else { //check other branches
			const KDtree::KDnode parent = tree[tree[best_index].parent];
			float parent_dist = fabs(dev_start[index][parent.split_axis] - dev_target[parent.current][parent.split_axis]);
			if (parent_dist < best_dist) {
				root = dev_start[index][parent.split_axis] < dev_target[parent.current][parent.split_axis] ? parent.right : parent.left;
				explored = true;
			}
			else {
				finished = true;
			}
		}
	}
	dev_cor[index] = tree[best_index].current;
}

__global__ void outerProduct(int n, glm::vec3 *dev_target, glm::vec3 *dev_start, int *dev_cor, glm::mat3 *product) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= n) {
		return;
	}
	 product[index] = glm::outerProduct(dev_target[dev_cor[index]], dev_start[index]);
}

__global__ void convergence(int startSize, glm::vec3 *dev_target, glm::vec3 *dev_start, int *cor, int *error) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= startSize) {
		return;
	}
	error[index] = glm::distance(dev_target[cor[index]], dev_start[index]);
}

void icpGPU(bool kdtree = false, bool checkConverge = false) {
	dim3 startblocksPerGrid((startSize + blockSize - 1) / blockSize);
	dim3 targetblocksPerGrid((targetSize + blockSize - 1) / blockSize);

	// mean center both data sets
	thrust::device_ptr<glm::vec3> thrust_target(dev_target);
	thrust::device_ptr<glm::vec3> thrust_start(dev_start);

	glm::vec3 targetMu = thrust::reduce(thrust_target, thrust_target + targetSize, glm::vec3(0.0f)) / float(targetSize);
	glm::vec3 startMu = thrust::reduce(thrust_start, thrust_start + startSize, glm::vec3(0.0f)) / float(startSize);

	transform << <targetblocksPerGrid, blockSize >> > (targetSize, dev_target, glm::mat3(1.0f), -targetMu);
	transform << <startblocksPerGrid, blockSize >> > (startSize, dev_start, glm::mat3(1.0f), -startMu);
	checkCUDAErrorWithLine("mean center failed!");

	// find correspondence
	if (kdtree) {
		correspondenceKDtree << <startblocksPerGrid, blockSize >> > (startSize, targetSize, dev_start, dev_target, dev_cor, dev_kd);
		checkCUDAErrorWithLine("Octree correspondences failed!");
	}
	else {
		correspondenceNaive << <startblocksPerGrid, blockSize >> > (startSize, targetSize, dev_start, dev_target, dev_cor);
		checkCUDAErrorWithLine("correspondences failed!");
	}

	// outer product of cor_target and dev_start for svd
	glm::mat3 *dev_M, U, S, V;
	cudaMalloc((void**)&dev_M, startSize * sizeof(glm::mat3));
	cudaMemset(dev_M, 0.0f, startSize * sizeof(glm::mat3));

	outerProduct << <startblocksPerGrid, blockSize >> > (startSize, dev_target, dev_start, dev_cor, dev_M);
	checkCUDAErrorWithLine("outer product  failed!");

	thrust::device_ptr<glm::mat3> thrust_M(dev_M);
	glm::mat3 M = thrust::reduce(thrust_M, thrust_M + startSize, glm::mat3(0.0f));

	// svd
	svd(M[0][0], M[1][0], M[2][0], M[0][1], M[1][1], M[2][1], M[0][2], M[1][2], M[2][2],
		U[0][0], U[1][0], U[2][0], U[0][1], U[1][1], U[2][1], U[0][2], U[1][2], U[2][2],
		S[0][0], S[1][0], S[2][0], S[0][1], S[1][1], S[2][1], S[0][2], S[1][2], S[2][2],
		V[0][0], V[1][0], V[2][0], V[0][1], V[1][1], V[2][1], V[0][2], V[1][2], V[2][2]
	);

	glm::mat3 I(1.0f);
	I[2][2] = glm::determinant(U*glm::transpose(V));
	// multiply for R, rotation
	glm::mat3 R = U * I * glm::transpose(V);
	glm::vec3 T = targetMu - R * startMu;

	// move start set
	transform << <startblocksPerGrid, blockSize >> > (startSize, dev_pos, R, T);
	cudaMemcpy(dev_start, dev_pos, startSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_target, &dev_pos[startSize], targetSize * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

	if (checkConverge) {
		int *dev_error;
		cudaMalloc((void**)&dev_error, startSize * sizeof(int));
		convergence <<< startblocksPerGrid, blockSize >>> (startSize, dev_target, dev_start, dev_cor, dev_error);

		thrust::device_ptr<int> thrust_error(dev_error);
		float error = thrust::reduce(thrust_error, thrust_error + startSize, 0.0f) / float(startSize);
		if (abs(error - prev_error) < ERROR) {
			converged = true;
			std::cout << "GPU average error: " << error << std::endl;
		}
		prev_error = error;
		cudaFree(dev_error);
	}

	cudaFree(dev_M);
	checkCUDAErrorWithLine("free memeory failed!");
}

void ICP::stepNaive(bool checkConverge) {
	icpGPU(false, checkConverge);
}

void ICP::stepKDtree(bool checkConverge) {
	icpGPU(true, checkConverge);
}

bool ICP::checkConvergence() {
	return converged;
}

void ICP::unitTest() {
	return;
}
