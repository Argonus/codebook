#include <string>    // string
#include <iostream>
#include <sstream>   // istringstream
#include <vector>    // vector
#include <algorithm> // reverse

using rowType = std::vector<int>;
using matrixType = std::vector<rowType>;

void printMatrix(matrixType matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        std::vector<int> row = matrix[i];

        for (int j = 0; j < row.size(); j++) {
            std::cout << " " << row[j];
        }

        std::cout << std::endl;
    }

}

void getMatrixRow(rowType& currentRowInt) {
    std::string rawRow;
    getline(std::cin, rawRow);

    std::istringstream ss(rawRow);
    std::string del;
    int idx = 0;

    while (getline(ss, del, ' ')) {
        currentRowInt[idx] = (std::stoi(del));
        idx++;
    }
}

void getMatrix(int rowsNum, int colsNum, matrixType& matrix) {
    for (unsigned short i = 0; i < rowsNum; i++) {
        std::string row;
        rowType matrixRow(colsNum);
        getMatrixRow(matrixRow);

        matrix[i] = matrixRow;
    }
}

void transposeMatrix(matrixType& matrix, matrixType& transpose) {
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[i].size(); j++)
        {
            transpose[j].push_back(matrix[i][j]);
        }
    }
}

matrixType multipleMatrices(const matrixType& matrixOne, const matrixType& matrixTwo)
{
    int rowsOne = matrixOne.size();
    int colsOne = matrixOne[0].size();
    int colsTwo = matrixTwo[0].size();

    matrixType outputMatrix(rowsOne, rowType(colsTwo, 0));

    for (int i = 0; i < rowsOne; ++i)
    {
        for (int j = 0; j < colsTwo; ++j)
        {
            for (int k = 0; k < colsOne; ++k)
            {
                outputMatrix[i][j] += matrixOne[i][k] * matrixTwo[k][j];
            }
        }
    }

    return outputMatrix;
}


int solve14()
{
    // base variables
    std::string value;
    int rows, columns;
    
    // get matrix size
    getline(std::cin, value);
    rows = std::stoi(value);
    getline(std::cin, value);
    columns = std::stoi(value);

    // get matrix
    matrixType matrix(rows);
    getMatrix(rows, columns, matrix);

    // calculate transpose
    matrixType transpose(columns);
    transposeMatrix(matrix, transpose);

    // calculate multi
    matrixType multi = multipleMatrices(transpose, matrix);
    printMatrix(multi);

    return 0;
}
