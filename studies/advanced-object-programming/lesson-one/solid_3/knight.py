from role import Role


class Knight(Role):
    name = "Knight"

    def __init__(self, weapon="", mount="", strength=1):
        self.weapon = weapon
        self.mount = mount
        self.strength = strength

    def describe(self):
        return "Fights using a {0}, rides on a {1}.".format(self.weapon, self.mount)

    def play(self):
        return self.fight()

    def fight(self):
        if self.strength <= 0:
            self.strength = 1.0
            return "   Resting..."
        self.strength -= 0.25
        return "Figting..."


def main():
    k = Knight("sword", "horse")
    print(k.describe())
    for i in range(6):
        print(k.fight())


if __name__ == "__main__":
    main()
