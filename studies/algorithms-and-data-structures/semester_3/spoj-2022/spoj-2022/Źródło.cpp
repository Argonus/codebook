#include <iostream>
#include <vector>
#include <tuple>

using relationPair = std::tuple<int, int>;
using relationsVector = std::vector<relationPair>;

using matrixRow = std::vector<int>;
using matrixType = std::vector<matrixRow>;

// 1 0 0 
// 0 1 0
// 0 0 1
bool isReflexive(matrixType matrix)
{
	for (int rc = 0; rc < matrix.size(); rc++) {
		if (matrix[rc][rc] == 0) { return false; }
	}

	return true;
}

// 0 1 1
// 1 0 1
// 1 1 0
bool isIreflexive(matrixType matrix)
{
	for (int rc = 0; rc < matrix.size(); rc++) {
		if (matrix[rc][rc] == 1) { return false; };
	}

	return true;
}

// 0 1 0
// 1 0 1
// 0 1 0
bool isSymmetric(matrixType matrix)
{
	for (int r = 0; r < matrix.size(); r++) {
		for (int c = r + 1; c < matrix.size(); c++) {
			if (matrix[r][c] != matrix[c][r]) { return false; };
		}
	}

	return true;
}

// 0 1 0
// 0 0 1
// 1 0 0
bool isAsymmetric(matrixType matrix, int maxNum) {
	for (int r = 0; r < maxNum - 1; r++) {
		for (int c = r + 1; c < maxNum; c++) {
			if (matrix[r][c] && matrix[c][r]) { return false; };
		}
	}

	return true;
}

bool isTransistive(matrixType matrix, int maxNum) {
	for (int r = 0; r < maxNum; r++) for (int c = 0; c < maxNum; c++) {
		int value = 0;

		for (int k = 0; (!value) && (k < maxNum); k++) {
			value = matrix[r][k] && matrix[k][c];
		}

		if (matrix[r][c] < value) { return false; };
	}

	return true;
}

bool isConnected(matrixType matrix, int maxNum) {
	for (int r = 0; r < maxNum - 1; r++) {
		for (int c = r + 1; c < maxNum; c++) {
			if (!(matrix[r][c] || matrix[c][r])) { return false; };
		}
	}

	return true;
}



int main()
{
	int numOne, numTwo;
	int maxNum = 0;
	short int maxRows = 100;
	relationsVector relations;

	// Build relations vector
	while (std::cin >> numOne >> numTwo) {
		if (maxRows < 0) { break; };

		if (numOne > maxNum) { maxNum = numOne; };
		if (numTwo > maxNum) { maxNum = numTwo; };
		relationPair pair = relationPair(numOne, numTwo);
		relations.push_back(pair);
		maxRows = maxRows - 1;
	}

	// Build binary matrix
	// 1 = true, 0 = false
	matrixType relationMatrix = matrixType(maxNum);

	// Build matrix with default 0
	for (int i = 0; i < maxNum; i++) {
		relationMatrix[i] = matrixRow(maxNum);
	}

	// Fill matrix with relations
	for (int i = 0; i < relations.size(); i++) {
		relationPair pair = relations[i];
		int rowIdx = std::get<0>(pair) - 1;
		int colIdx = std::get<1>(pair) - 1;

		relationMatrix[rowIdx][colIdx] = 1;
	}

	bool any = false;
	bool ref = false, iref = false, sym = false, asym = false, tran = false, conn = false;

	// Print line one
	if (isReflexive(relationMatrix)) { std::cout << "Z" << " "; ref = true; any = true; };
	if (isIreflexive(relationMatrix)) { std::cout << "PC" << " "; iref = true; any = true; };
	if (isSymmetric(relationMatrix)) { std::cout << "S" << " "; sym = true; any = true; };
	if (isAsymmetric(relationMatrix)) { std::cout << "AS" << " "; asym = true; any = true; };
	if (isTransistive(relationMatrix)) { std::cout << "P" << " "; tran = true; any = true; };
	if (isConnected(relationMatrix)) { std::cout << "SP" << " "; conn = true; any = true; };
	if (!any) { std::cout << "X"; return 0; }
	// Print line two
	std::cout << "\n";
	bool anyTwo = false;
	if (ref && sym && tran) { std::cout << "RR" << " "; anyTwo = true; };
	if (ref && asym && tran && conn) { std::cout << "RPL" << " "; anyTwo = true; };
	if (ref && asym && tran) { std::cout << "RPCz" << " "; anyTwo = true; };
	if (!anyTwo) { std::cout << "X"; };

	return 0;
}