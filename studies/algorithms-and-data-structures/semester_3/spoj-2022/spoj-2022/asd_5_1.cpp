#include <iostream>
#include <vector>

using numbersVector = std::vector<int>;

void getInput(numbersVector& numbers) {
	int number;
	while (std::cin >> number) {
		numbers.push_back(number);
	}
}

void printVector(numbersVector& numbers) {
	for (int i = 0; i < numbers.size(); ++i) {
		std::cout << numbers[i] << "\n";
	}
}

void quickSort(numbersVector& numbers, int startIdx, int endIdx)
{
    int i = startIdx;
    int j = endIdx;
    int pivotIdx = (startIdx + endIdx) / 2;
    int pivot = numbers[pivotIdx];

    do {
        while (numbers[i] < pivot) { i++; };
        while (numbers[j] > pivot) { j--; };

        if (i <= j) {
            std::swap(numbers[i], numbers[j]);
            i++;
            j--;
        }
    } while (i <= j);

    if (startIdx < j) quickSort(numbers, startIdx, j);
    if (endIdx > i) quickSort(numbers, i, endIdx);
}

int solve51() {
	numbersVector numbers;

	getInput(numbers);
	quickSort(numbers, 0, numbers.size() - 1);
	printVector(numbers);

	return 0;
}