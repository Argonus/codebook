#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

using WordSet = std::vector<std::string>;
using LengthMatrix = std::vector<std::vector<int>>;
const int maxSize = 1500;

int minFromThree(int intOne, int intTwo, int intThree)
{
	int min = intOne < intTwo ? intOne : intTwo;
	return min < intThree ? min : intThree;
}

int calcDistance(const std::string& wordOne, const std::string& wordTwo)
{
    int lengthOne = wordOne.length();
    int lenghtTwo = wordTwo.length();
    
    if (lengthOne == 0) { return lenghtTwo; }
    if (lenghtTwo == 0) { return lengthOne; }

    std::vector<int> calculations(lenghtTwo + 1);
    
    for (int i = 0; i < lenghtTwo + 1; i++) {
        calculations[i] = i;
    }

    for (int i = 0; i < lengthOne; i++) {
        calculations[0] = i + 1;
        int v1 = i;

        for (int j = 0; j < lenghtTwo; j++) {
            int v2 = calculations[j + 1];

            if (wordOne[i] == wordTwo[j])
            {
                calculations[j + 1] = v1;
            }
            else
            {
                int v3 = calculations[j];
                int temp = v2 < v1 ? v2 : v1;
                int min = temp < v3 ? temp : v3;
                calculations[j + 1] = min + 1;
            }

            v1 = v2;
        }
    }

    return calculations[lenghtTwo];
}

void collectWords(WordSet& words)
{
	std::string word;
	while (std::cin >> word) {
        if (word == "okon") { break; }
        
		words.push_back(word);
	}
}
    
void printWordsLength(const WordSet& words) {
    for (int idx = 0; idx < words.size(); idx++) {
        int wordNum = idx + 1;
		std::cout << wordNum << ": " << words.at(idx) << " - " << words.at(idx).length() << std::endl;
    }
}

void buildDistanceMatrix(const WordSet& words, LengthMatrix& matrix) {
	for (int rowNum = 0; rowNum < words.size(); rowNum++) {
		std::vector<int> row;
        
		for (int colNum = 0; colNum < words.size(); colNum++) {
			std::string wordOne = words[rowNum];
			std::string wordTwo = words[colNum];            

			int distance = calcDistance(wordOne, wordTwo);
			row.push_back(distance);
		}
        
		matrix.push_back(row);
	}
}

void printDistanceMatrixHeader(const WordSet& words) {
	// Line One Header
    std::cout << "x |";
    for (int idx = 0; idx < words.size(); idx++) {
		int wordNum = idx + 1;
        std::cout << " " << wordNum;
    }
	// Line Two Header
    std::cout << std::endl;
    std::cout << "- +";
    for (int i = 0; i < words.size(); i++) {
        std::cout << " " << "-";
    }
    std::cout << std::endl;
}

void printDistanceMatrix(const LengthMatrix& matrix) {
	for (int rowNum = 0; rowNum < matrix.size(); rowNum++) {
		int wordNum = rowNum + 1;
		std::cout << wordNum << " |";
		for (int colNum = 0; colNum < matrix.size(); colNum++) {
			int distance = matrix[rowNum][colNum];
			std::cout << " " << distance;
		}
		std::cout << std::endl;
	}
}

void printProbabilityMatrix(WordSet& words, LengthMatrix& matrix) {
    int size = words.size();
    for (int row = 0; row < size; row++) {
        for (int col = 0; col < size; col++) {
            float dist = matrix[row][col];
            float max = std::max(words[row].length(), words[col].length());
            float probability = 1 - (dist / max);

            std::cout << std::fixed << probability << " ";
        }
        std::cout << std::endl;
    }
}

int solve72()
{   
	WordSet words;
    
    // Part One
    collectWords(words);
	printWordsLength(words);
    std::cout << std::endl;
    // Part Two
    std::vector<std::vector<int>> matrix;
	buildDistanceMatrix(words, matrix);
    printDistanceMatrixHeader(words);
    printDistanceMatrix(matrix);
    std::cout << std::endl;
    // Part Three
    std::cout.precision(3);
    printProbabilityMatrix(words, matrix);

    return 0;
}