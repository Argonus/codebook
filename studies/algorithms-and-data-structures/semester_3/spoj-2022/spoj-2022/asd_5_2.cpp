#include <iostream>
#include <vector>
#include <algorithm>

using rowType = std::vector<double>;
using matrixType = std::vector<rowType>;
using sortIdx = std::vector<int>;

void printMatrix(matrixType matrix) {
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}

}

void parseInput(matrixType& dataMatrix, int colsNum) {
	double number;

	while (std::cin >> number) {
		rowType dataRow;
		dataRow.push_back(number);

		// First has been already fetched
		for (int i = 0; i < colsNum - 1; i++) {
			std::cin >> number;
			dataRow.push_back(number);
		};

		dataMatrix.push_back(dataRow);
	}
}

// Prepare struct to store columns to sort idx
struct RowCompare {
	RowCompare(sortIdx colsIdx) { this->colsIdx = colsIdx; };
	sortIdx colsIdx;

	bool operator() (rowType& leftRow, rowType& rightRow) { 
		// Default is false.
		bool result = false;

		for (int i = 0; i < colsIdx.size(); i++) {
			int colIdx = colsIdx[i];

			if (leftRow[colIdx] == rightRow[colIdx]) {
				continue;
			}
			else {
				result = (leftRow[colIdx] < rightRow[colIdx]);
				break;
			}
		};

		return result;
	}
};

void sortMatrix(matrixType& dataMatrix, sortIdx colsIdx) {
	std::sort(dataMatrix.begin(), dataMatrix.end(), RowCompare(colsIdx));
}

int solve52() {
	int colsNum;
	int sortableColsNum;
	sortIdx sortableColsIdx;

	// Input Metadata
	std::cin >> colsNum;
	std::cin >> sortableColsNum;
	for (int i = 0; i < sortableColsNum; i++) {
		int num, colIdx;
		std::cin >> num;
		colIdx = num - 1;
		sortableColsIdx.push_back(colIdx);
	}
	std::reverse(sortableColsIdx.begin(), sortableColsIdx.end());

	// Parse Input
	matrixType dataMatrix;
	parseInput(dataMatrix, colsNum);

	// Sort Input Params
	sortMatrix(dataMatrix, sortableColsIdx);
	printMatrix(dataMatrix);

	return 0;
}