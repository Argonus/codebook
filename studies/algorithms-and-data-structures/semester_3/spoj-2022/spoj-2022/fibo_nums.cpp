#include <string>
#include <iostream>
#include <map>

const long moduloVal = 1000000007;
// Map to keep results. Baisc calculation doesnt work with large numbers
using number = unsigned long long;
std::map<number, number> results;

// Instead of calculating fibonacci number, we already calculate it modulo
// Memoize results in map ensures no double calculations
number fibonacciModulo(number n) {
	if (n < 2) return 1;

	try {
		return results.at(n);
	}
	catch (std::out_of_range) {
		number n1 = fibonacciModulo((n + 1) / 2);
		number n2 = fibonacciModulo(n / 2);
		number n3 = fibonacciModulo((n - 1) / 2);
		number n4 = fibonacciModulo((n - 2) / 2);

		results[n] = (n1 * n2 + n3 * n4) % moduloVal;
		return results[n];
	};
}

int fiboNums() {
	std::string value;
	getline(std::cin, value);
	short int testsNumber = std::stoi(value);

	for (short int i = 0; i < testsNumber; i++) {
		getline(std::cin, value);
		number currentNumber = std::stoll(value) - 1;
		number fibonacciNumber = fibonacciModulo(currentNumber);

		std::cout << fibonacciNumber << std::endl;
	}

	return 0;
}