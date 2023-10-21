#include <string>    
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using rowType = std::vector<int>;
using matrixType = std::vector<rowType>;
using shortInt = signed short int;


matrixType buildMovementMatrix(shortInt vectorsNum, shortInt spaceSize) {
	matrixType movementMatrix(vectorsNum);

	for (shortInt i = 0; i < vectorsNum; i++)
	{
		rowType movementRow;
		for (shortInt j = 0; j < spaceSize; j++)
		{
			shortInt num;
			cin >> num;
			movementRow.push_back(num);
		}

		movementMatrix[i] = movementRow;
	}

	return movementMatrix;
}

bool isZeroVector(rowType movementVector)
{
	return all_of(movementVector.begin(), movementVector.end(), [](shortInt i) { return i == 0; });
}

bool isPerpendicularVector(rowType oldMovementVector, rowType newMovementVector)
{
	shortInt product = 0;
	bool result;

	for (shortInt i = 0; i < oldMovementVector.size(); i++) {
		product += oldMovementVector[i] * newMovementVector[i];
	};

	result = (product == 0) ? true : false;
	return result;
}

rowType addToVector(rowType position, rowType vector)
{
	for (int i = 0; i < position.size(); i++) {
		position[i] += vector[i];
	}

	return position;
}

rowType traverseRoute(matrixType movementMatrix) {
	// First vector should be good. We will see in tests :D 
	rowType goodMovementVector = movementMatrix[0];
	// First movement always succeded, no need to store 0,0...0 vector
	rowType actualPosition = goodMovementVector;

	for (int i = 1; i < movementMatrix.size(); i++)
	{
		rowType newMovementVector = movementMatrix[i];

		if (isZeroVector(newMovementVector)) { continue; };
		if (!isPerpendicularVector(goodMovementVector, newMovementVector)) { continue; };

		actualPosition = addToVector(actualPosition, newMovementVector);
		goodMovementVector = newMovementVector;
	}

	return actualPosition;
}

void displayPosition(rowType position)
{
	for (int i = 0; i < position.size(); i++) {
		cout << position[i] << " ";
	}
	cout << "\n";
}

int solve21() {
	unsigned short int dataSetsNum;
	cin >> dataSetsNum;

	for (int i = 0; i < dataSetsNum; i++) {
		// Loop Vars
		shortInt spaceSize, vectorsNum;
		// Description input
		cin >> spaceSize;
		cin >> vectorsNum;
		matrixType movementMatrix = buildMovementMatrix(vectorsNum, spaceSize);
		rowType lastPosition = traverseRoute(movementMatrix);
		displayPosition(lastPosition);
	}

	return 0;
};
