#include <string>
#include <iostream>

int run_asd_1_1()
{
    std::string value;
    int result = 0;


    while (getline(std::cin, value)) {
        int current = std::stoi(value);

        if (current == 0) {
            std::cout << result;
            break;
        };

        if (result < current) {
            result = current;
        };
    }

    return 0;
}