#include <iostream>
#include <string>
#include <vector>
#include <cmath>

struct Student {
	int applicationOrder;
	float studentNumber;
	std::string name;
	std::string surname;
};
using studentsList = std::vector<Student>;
using operatorFunction = bool (const Student&, const Student&);

bool byNumber(const Student& a, const Student& b) {
	float numA = ::floorf(a.studentNumber * 1000000) / 1000000;
	float numB = ::floorf(b.studentNumber * 1000000) / 1000000;
	return numA <= numB;
}

bool byName(const Student& a, const Student& b) {
	return a.name <= b.name;
}

bool bySurname(const Student& a, const Student& b) {
	return a.surname <= b.surname;
}

void loadList(studentsList& students) {
	int applicationOrder;

	while (std::cin >> applicationOrder) {
		Student s;
		s.applicationOrder = applicationOrder;
		std::cin >> s.studentNumber;
		std::cin >> s.name;
		std::cin >> s.surname;
		students.push_back(s);
	}
}

void merge(studentsList& students, int left, int mid, int right, operatorFunction func) {
	studentsList temp;
	int i, j;
	i = left;
	j = mid + 1;

	while (i <= mid && j <= right) {
		if (func(students[i], students[j])) {
			temp.push_back(students[i]);
			++i;
		}
		else {
			temp.push_back(students[j]);
			++j;
		}

	}

	while (i <= mid) {
		temp.push_back(students[i]);
		++i;
	}

	while (j <= right) {
		temp.push_back(students[j]);
		++j;
	}

	for (int i = left; i <= right; ++i)
		students[i] = temp[i - left];

}

void mergeSort(studentsList& students, int left, int right, operatorFunction func) {
	if (left < right) {
		int mid = (left + right) / 2;
		mergeSort(students, left, mid, func);
		mergeSort(students, mid + 1, right, func);
		merge(students, left, mid, right, func);
	}
}

void mergeSort(studentsList& students, operatorFunction func) {
	mergeSort(students, 0, students.size() - 1, func);
}

void sortList(studentsList& students) {
	mergeSort(students, byNumber);
	mergeSort(students, byName);
	mergeSort(students, bySurname);
}

void printList(studentsList& students) {
	for (int i = 0; i < students.size(); i++) {
		int appOrd = students[i].applicationOrder;
		double studNum = students[i].studentNumber;
		std::string name = students[i].name;
		std::string surname = students[i].surname;
		
		std::cout << i + 1 << " " << surname << " " << name << " " << studNum << " " << appOrd << std::endl;
	}
}

int solve53() {
	studentsList students;
	loadList(students);
	sortList(students);
	printList(students);
	return 0;
}