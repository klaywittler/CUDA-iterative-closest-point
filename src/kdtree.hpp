#pragma once

#include <vector>
#include "glm/glm.hpp"

namespace KDtree {
	template<typename T>
	std::vector<T> arange(T start, T stop, T step = 1) {
		std::vector<T> values;
		for (T value = start; value < stop; value += step)
			values.push_back(value);
		return values;
	}

	enum Axis {
		X = 0, Y, Z
	};

	struct KDnode {
		KDnode();
		KDnode(int parentIndex, int currentIndex, Axis axis);

		int parent;
		int current;
		int left;
		int right;
		Axis split_axis;

		int neighbors[6];
		bool isLeaf;
	};

	void buildTree(const std::vector<glm::vec3> &input, KDnode *tree, int parent = -1, int index = 0, std::vector<int> inputIdx = std::vector<int>());
}
