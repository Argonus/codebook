#include <string>
#include <iostream>

int solve13()
{
    std::string value;
    signed int resultMax;
    signed int resultMin;
    signed int current;

    getline(std::cin, value);
    current = std::stoi(value);
    resultMax = current;
    resultMin = current;

    unsigned int resultMaxIdx = 1;
    unsigned int resultMinIdx = 1;
    unsigned int currentIdx = 1;

    while (getline(std::cin, value)) {
        if (value == "stop") {
            break;
        }

        current = std::stoi(value);
        currentIdx = currentIdx + 1;

        if (resultMin > current) {
            resultMin = current;
            resultMinIdx = currentIdx;
        };

        if (resultMax <= current) {
            resultMax = current;
            resultMaxIdx = currentIdx;
        }
    }

    std::cout << resultMin;
    std::cout << "\n";
    std::cout << resultMinIdx;
    std::cout << "\n";
    std::cout << resultMax;
    std::cout << "\n";
    std::cout << resultMaxIdx;

    return 0;
}
