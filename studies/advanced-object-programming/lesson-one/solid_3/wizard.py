from role import Role


class Wizard(Role):
    name = "Wizard"

    def __init__(self, mana=3):
        self.mana = mana

    def describe(self):
        return "Casts spells, {0} mana left.".format(self.mana)

    def play(self):
        return self.cast_a_spell()

    def cast_a_spell(self):
        if self.mana <= 0:
            self.mana = 3
            return "Meditating to restore mana..."
        self.mana -= 1
        return "Casting a spell..."


def main():
    w = Wizard()
    print(w.describe())
    for i in range(6):
        print(w.cast_a_spell())


if __name__ == "__main__":
    main()
