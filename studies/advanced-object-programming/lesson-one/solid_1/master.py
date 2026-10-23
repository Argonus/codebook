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
        self.heros = []

    def show(self):
        print("\nThe heros: ")
        for h in self.heros:
            print(h.describe())

    def play(self):
        for i in range(10):
            hero = choice(self.heros);
            print("\nThe working hero:", hero.describe())
            while True:
                if hero.name_of_the_role == "Knight":  # the inhabitant is a Knight
                    print(hero.fight())
                elif hero.name_of_the_role == "Gardener":  # the inhabitant is a Gardener
                        print(hero.plant())
                if uniform(0, 1) >= 0.75:
                    break

    def set_hero(self, hero):
        self.heros.append(hero)


def main():
    m = Master()
    m.set_hero(Gardener("Slenderwillow","f","Oldoak","Sunnydaisy"))
    m.set_hero(Knight("Quickhand","m","Bravedeed","Sunbell","sword","horse"))
    m.set_hero(Knight("Truefriend","m","Highspirit", "Mistymorning","spear","dragon"))
    m.set_hero(Gardener("Greenleaf","m","Brownleaf","Goldendaisy"))
    m.show()
    m.play()

if __name__ == "__main__":
    main()

