#include <iostream>
#include <array>
#include <string>


using stackType = std::array<int, 10>;

void addToStack(stackType& stack, short int& counter) {
    if (counter < 10) {
        std::cin >> stack.at(counter);
        counter++;
        std::cout << ":)" << "\n";
    }
    else {
        std::cout << ":(" << "\n";
    }
}

void removeFromStack(stackType& stack, short int& counter) {
    if (counter > 0) {
        int val = stack[counter - 1];
        counter--;
        std::cout << val << "\n";
    }
    else {
        std::cout << ":(" << "\n";
    }
}

int solveStos() {
	char sign;
    short int counter = 0;
    stackType stack;

    while (std::cin >> sign) {
        switch (sign) {
            case '+':
                addToStack(stack, counter);
                break;
            case '-':
                removeFromStack(stack, counter);
                break;
        }
    }

	return 0;
}