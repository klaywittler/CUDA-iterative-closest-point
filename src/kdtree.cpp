#pragma once

#include <vector>
#include <algorithm>
#include "glm/glm.hpp"
#include "kdtree.hpp"

KDtree::KDnode::KDnode() {
	current = -1;
	parent = -1;
	left = -1;
	right = -1;
	split_axis = X;
	isLeaf = true;
}

KDtree::KDnode::KDnode(int parentIndex, int currentIndex, Axis axis) {
	parent = parentIndex;
	current = currentIndex;
	left = -1;
	right = -1;
	split_axis = axis;
	isLeaf = true;
}

void KDtree::buildTree(const std::vector<glm::vec3> &input, KDnode *tree, int parent, int index, std::vector<int> inputIdx) {
	Axis split_axis;
	if (inputIdx.size() == 0 && parent == -1 && index == 0) {
		split_axis = X;
		inputIdx = arange<int>(0, input.size());
	}
	else {
		split_axis = Axis((tree[parent].split_axis + 1) % 3);
	}

	auto lambda = [&input, &split_axis](const int &p, const int &q) { return input[p][split_axis] < input[q][split_axis]; };
	std::sort(inputIdx.begin(), inputIdx.end(), lambda);

	int middle = inputIdx.size() / 2;
	tree[index] = KDnode(parent, inputIdx[middle], split_axis);

	if (middle > 0) {
		tree[index].isLeaf = false;
		tree[index].left = index + 1;
		std::vector<int> leftTree(inputIdx.begin(), inputIdx.begin() + middle);
		buildTree(input, tree, index, tree[index].left, leftTree);
	}
	if (middle < inputIdx.size() - 1) {
		tree[index].isLeaf = false;
		tree[index].right = index + middle + 1;
		std::vector<int> rightTree(inputIdx.begin() + middle + 1, inputIdx.end());
		buildTree(input, tree, index, tree[index].right, rightTree);
	}
}

