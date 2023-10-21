#include <string>
#include <iostream>

int solve12()
{
    std::string value;
    signed int result;
    unsigned int resultIdx = 1;

    getline(std::cin, value);
    result = std::stoi(value);
    unsigned int idx = 1;

    while (getline(std::cin, value)) {
        int current = std::stoi(value);
        idx = idx + 1;

        if (current == 0) {
            std::cout << result;
            std::cout << "\n";
            std::cout << resultIdx;
            break;
        };

        if (result >= current) {
            result = current;
            resultIdx = idx;
        };
    }

    return 0;
}