import random

from role import Role


class Gardener(Role):
    name = "Gardener"
    NICE_WEATHER_PROBABILITY = 0.75

    def __init__(self):
        self.random_generator = random.Random()
        self.number_of_planted_trees = 0

    def describe(self):
        return "Planted {0} trees.".format(self.number_of_planted_trees)

    def play(self):
        return self.plant()

    def plant(self):
        if self.random_generator.random() < self.NICE_WEATHER_PROBABILITY:
            self.number_of_planted_trees += 1
            return "Planting a tree..."
        return "Looking at trees growing in the rain..."


def main():
    g = Gardener()
    print(g.describe())
    for i in range(6):
        print(g.plant())


if __name__ == "__main__":
    main()
