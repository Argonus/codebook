#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>


using vectorType = std::vector<double>;
using rowType = std::vector<double>;
using matrixType = std::vector<rowType>;

void captureInput(int numRows, int numColumns, matrixType& matrix, vectorType& vector) {
	for (int row = 0; row < numRows; row++) {
		rowType matrixRow = rowType(numColumns);

		for (int col = 0; col < numColumns; col++) {
			// firs n elements are matrix values
			double val;
			std::cin >> val;
			matrixRow[col] = val;
		}
		matrix[row] = matrixRow;

		double vectorVal;
		std::cin >> vectorVal;
		vector[row] = vectorVal;
	}
}

void transposeMatrix(matrixType& matrix, matrixType& transpose) {
	int matrixSize = matrix.size();
	int rowSize = matrix[0].size();

	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0; j < rowSize; j++) {
			transpose[j].push_back(matrix[i][j]);
		}
	}
}

void multiplyMatrix(matrixType& transposedMatrix, vectorType& vector, vectorType& resultVector) {
	int matrixSize = transposedMatrix.size();
	int rowSize = transposedMatrix[0].size();

	for (int row = 0; row < matrixSize; row++) {
		for (int col = 0; col < rowSize; col++) {
			resultVector[row] += transposedMatrix[row][col] * vector[col];
		}
	}
}

void printVector(vectorType& resultVector) {
	int vectorSize = resultVector.size();
	std::string output;

	for (int col = 0; col < vectorSize; col++) {
		double value = resultVector[col];
		printf("%.6f ", value);
	}
}

int solve41() {
	int numColumns;
	std::cin >> numColumns;
	int numRows;
	std::cin >> numRows;

	// Init values
	vectorType vector(numRows);
	matrixType matrix(numRows);
	matrixType transposedMatrix(numColumns);
	// Result is always vector
	vectorType resultVector(numColumns);

	captureInput(numRows, numColumns, matrix, vector);
	transposeMatrix(matrix, transposedMatrix);
	multiplyMatrix(transposedMatrix, vector, resultVector);
	printVector(resultVector);

	return 0;
};