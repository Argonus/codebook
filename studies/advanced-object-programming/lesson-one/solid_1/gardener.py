# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:57:23 2019

@author: Ewa
"""


import random


class Gardener:
    NICE_WEATHER_PROBABILITY = 0.75

    def __init__(self, name, gender, father_name, mother_name):
        self.name = name
        self.gender = gender
        self.father_name = father_name
        self.mother_name = mother_name
        self.number_of_planted_trees = 0
        self.random_generator = random.Random()  # instantiation of rand. gen.
        self.name_of_the_role = "Gardener"

    def describe(self):
         return "{0} {1}, {2} of {3} and {4}. Planted {5} trees. ".format(
              self.name_of_the_role,
              self.name, "son" if self.gender == "m" else "daughter",
              self.father_name, self.mother_name, self.number_of_planted_trees)

    def plant(self):
        if self.random_generator.random() < self.NICE_WEATHER_PROBABILITY:
            self.number_of_planted_trees += 1
            return "Planting a tree..."
        return "Looking at trees growing in the rain..."


def main():
    k = Gardener("Greenleaf","m","Brownleaf","Goldendaisy")
    print(k.describe())
    for i in range(6):
        print(k.plant())


if __name__ == "__main__":
    main()
