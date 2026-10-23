# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:57:32 2019

@author: Ewa
"""


from random import choice, uniform
from gardener import Gardener
from knight import Knight


class Master:
    def __init__(self):
        self.persons = []

    def show(self):
        print("\nPersons: ")
        for p in self.persons:
            print(p.describe())

    def play(self):
        for i in range(10):
            person = choice(self.persons);
            print("\nWorking person:", person.describe())
            while True:
                if person.get_role() == "Knight":
                    print(person.fight())
                else:
                    print(person.plant())
                if uniform(0, 1) >= 0.75:
                    break

    def set_person(self, person):
        self.persons.append(person);


def main():
    m = Master()
    m.set_person(Gardener("Greenleaf", "m", "Brownleaf", "Goldendaisy"))
    m.set_person(Gardener("Slenderwillow", "f", "Oldoak", "Sunnydaisy"))
    m.set_person(Knight("Quickhand", "m", "Bravedeed","Sunbell", "sword", "horse"))
    m.set_person(Knight("Truefriend", "m", "Highspirit","Mistymorning", "spear", "dragon"))
    m.show()
    m.play()

if __name__ == "__main__":
    main()

