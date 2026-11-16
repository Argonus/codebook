from random import choice, uniform


class Master:
    def __init__(self):
        self.persons = []

    def set_person(self, person):
        self.persons.append(person)

    def show(self):
        print("\nPersons: ")
        for p in self.persons:
            print(p.describe())

    def play(self):
        for i in range(10):
            person = choice(self.persons)
            print("\nWorking person:", person.describe())
            while True:
                print(person.play())
                if uniform(0, 1) >= 0.75:
                    break

    def change_role(self, name, new_role):
        self.find(name).change_role(new_role)

    def find(self, name):
        for p in self.persons:
            if p.name == name:
                return p
        raise LookupError("No inhabitant named " + name)


def main():
    from gardener import Gardener
    from knight import Knight
    from person import Person
    from wizard import Wizard

    m = Master()
    m.set_person(Person("Greenleaf", "m", "Brownleaf", "Goldendaisy", Gardener()))
    m.set_person(Person("Slenderwillow", "f", "Oldoak", "Sunnydaisy", Gardener()))
    m.set_person(Person("Quickhand", "m", "Bravedeed", "Sunbell", Knight("sword", "horse")))
    m.set_person(Person("Truefriend", "m", "Highspirit", "Mistymorning", Knight("spear", "dragon")))
    m.show()
    m.play()
    m.change_role("Quickhand", Wizard())
    m.show()
    m.play()


if __name__ == "__main__":
    main()
