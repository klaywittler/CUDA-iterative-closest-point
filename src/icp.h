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
    void initSimulation(std::vector<glm::vec3> start, std::vector<glm::vec3> target, bool transformScan);
    void copyToVBO(float *vbodptr_positions, float *vbodptr_velocities);
	void stepCPU(bool checkConverge = false);
	void stepNaive(bool checkConverge = false);
	void stepKDtree(bool checkConverge = false);
	bool checkConvergence();
    void endSimulation();
    void unitTest();
}
