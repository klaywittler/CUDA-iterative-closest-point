#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace ICP {
    void initSimulation(std::vector<glm::vec3> start, std::vector<glm::vec3> target);
    void copyToVBO(float *vbodptr_positions, float *vbodptr_velocities);
	void stepCPU();
	void stepGPU();
    void endSimulation();
    void unitTest();
}
