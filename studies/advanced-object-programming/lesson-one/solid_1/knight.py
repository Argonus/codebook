# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:57:12 2019

@author: Ewa
"""


class Knight:
    def __init__(self, name, gender, fatherName, motherName, weapon="", mount="", strength=1):
        self.name = name
        self.gender = gender
        self.fatherName = fatherName
        self.motherName = motherName
        self.weapon = weapon
        self.mount = mount
        self.strength = strength
        self.name_of_the_role = "Knight"

    def describe(self):
        return "{0} {1}, {2} of {3} and {4}. Fights using a {5} rides on a {6}. ".format(
              self.name_of_the_role,self.name, "son" if self.gender == "m" else "daugher",
              self.fatherName, self.motherName, self.weapon, self.mount,
              "Can fight" if self.strength > 0 else "Needs a rest")

    def fight(self):
        if self.strength <= 0:
            self.strength = 1.0
            return "   Resting ..."
        self.strength -= 0.25
        return "Figting..."


def main():
    k = Knight("Quickhand", "m", "Bravedeed", "Sunbell", "sword", "horse")
    print(k.describe())
    for i in range(12):
        print(k.fight())


if __name__ == "__main__":
    main()
