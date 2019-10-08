#include <iostream>
#include "pointCloud.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "utilityCore.hpp"

PointCloud::PointCloud(string filename) {
    cout << "Reading PointCloud from " << filename << " ..." << endl;
    cout << " " << endl;
	float factor = 1.0f;
	if (filename.find("bunny") != std::string::npos) {
		factor = 500.0f;
	}
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
			glm::vec3 p(atof(tokens[0].c_str()), atof(tokens[1].c_str()), atof(tokens[2].c_str()));
			points.push_back(factor*p);
        }
    }
}


