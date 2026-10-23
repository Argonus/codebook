# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:57:40 2019

@author: Ewa
"""

from gardener import Gardener
from knight import Knight
from master import Master


def main():
    print("\n\nSOLID exercise 1\n")
    master = Master()
    master.set_person(Gardener("Greenleaf", "m", "Brownleaf", "Goldendaisy"))
    master.set_person(Gardener("Slenderwillow", "f", "Oldoak", "Sunnydaisy"))
    master.set_person(Knight("Quickhand", "m", "Bravedeed","Sunbell", "sword", "horse"))
    master.set_person(Knight("Truefriend", "m", "Highspirit","Mistymorning", "spear", "dragon"))
    print("\nList of inhabitants:\n")
    master.show()
    print("\nPlaying:\n")
    master.play()


if __name__ == "__main__":
    main()
