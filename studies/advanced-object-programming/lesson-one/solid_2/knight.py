# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:57:12 2019

@author: Ewa
"""


from person import Person

class Knight(Person):
    def __init__(self, name, gender, father_name, mother_name, weapon="", mount="", strength=1):
        super().__init__(name, gender, father_name, mother_name)
        self.weapon = weapon
        self.mount = mount
        self.strength = strength
        self.role = "Knight"

    def describe(self):
        return "Knight {0}, {1} of {2} and {3}. Fights using a {4} rides on a {5}. ".format(
              self.name, "son" if self.gender == "m" else "daugher",
              self.father_name, self.mother_name, self.weapon, self.mount,
              "Can fight" if self.strength > 0 else "Needs a rest")

    def fight(self):
        if self.strength <= 0:
            self.strength = 1.0
            return "   Resting..."
        self.strength -= 0.25
        return "Figting..."


def main():
    k = Knight("Quickhand", "m", "Bravedeed", "Sunbell", "sword", "horse")
    print(k.describe())
    for i in range(6):
        print(k.fight())


if __name__ == "__main__":
    main()
