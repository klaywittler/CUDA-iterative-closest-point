#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"

using namespace std;

class PointCloud {
private:
    ifstream fp_in;
public:
	PointCloud(string filename);
    ~PointCloud();

    std::vector<glm::vec3> points;
};
