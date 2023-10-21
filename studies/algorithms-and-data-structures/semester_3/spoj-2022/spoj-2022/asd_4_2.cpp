#include <iostream>
#include <string>
#include <array>
#include <vector>

const int maxQueueSize = 10;
using queueType = std::array<int, maxQueueSize>;

void enqueue(queueType& queue, int& queueSize) {
	int number;
	std::cin >> number;

	if (queueSize < maxQueueSize) {
		std::cout << "--->" << std::endl;
		queue[queueSize] = number;
		queueSize++;
	}
	else {
		std::cout << "Error: queue is full" << std::endl;
	}
}

void dequeue(queueType& queue, int& queueSize) {
	if (queueSize > 0) {
		std::cout << queue[0] << std::endl;
		// Remove first element
		for (int i = 0; i < queueSize - 1; i++) {
			queue.at(i) = queue.at(i + 1);
		}

		queueSize--;
	}
	else {
		std::cout << "Error: queue is empty" << std::endl;
	}
}

void print(queueType& queue, int& queueSize) {
	if (queueSize > 0) {
		std::string output = "Print: ";

		for (int i = 0; i < queueSize; i++)
		{
			output += std::to_string(queue.at(i));
			output += " ";
		}

		std::cout << output << std::endl;
	}
	else {
		std::cout << "Print: Queue is empty" << std::endl;
	}
}

const std::string enqueueCmd = "Enqueue";
const std::string dequeueCmd = "Dequeue";
const std::string printCmd = "Print";

int solve42() {
	queueType queue;
	std::string command;
	int queueSize = 0;

	while (std::cin >> command) {
		if (command == enqueueCmd) { 
			enqueue(queue, queueSize); 
			continue; 
		}

		if (command == dequeueCmd) { 
			dequeue(queue, queueSize); 
			continue; 
		}

		if (command == printCmd) { 
			print(queue, queueSize); 
		}
	}

	return 0;
}