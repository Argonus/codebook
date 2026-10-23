class Person:
    def __init__(self, name, gender, father_name, mother_name, role):
        self.name = name
        self.gender = gender
        self.father_name = father_name
        self.mother_name = mother_name
        self.role = role

    def describe(self):
        return "{0} {1}, {2} of {3} and {4}. {5}".format(
            self.role.name, self.name, "son" if self.gender == "m" else "daugher",
            self.father_name, self.mother_name, self.role.describe())

    def play(self):
        return self.role.play()

    def change_role(self, new_role):
        self.role = new_role


def main():
    from knight import Knight
    from wizard import Wizard

    p = Person("Quickhand", "m", "Bravedeed", "Goldendaisy", Knight("sword", "horse"))
    print(p.describe())
    print(p.play())
    p.change_role(Wizard())
    print(p.describe())
    print(p.play())


if __name__ == "__main__":
    main()
