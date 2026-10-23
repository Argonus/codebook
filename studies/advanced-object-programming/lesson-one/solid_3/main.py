from gardener import Gardener
from knight import Knight
from master import Master
from person import Person
from wizard import Wizard


def main():
    print("\n\nSOLID exercise 4\n")
    master = Master()
    master.set_person(Person("Greenleaf", "m", "Brownleaf", "Goldendaisy", Gardener()))
    master.set_person(Person("Slenderwillow", "f", "Oldoak", "Sunnydaisy", Gardener()))
    master.set_person(Person("Quickhand", "m", "Bravedeed", "Sunbell", Knight("sword", "horse")))
    master.set_person(Person("Truefriend", "m", "Highspirit", "Mistymorning", Knight("spear", "dragon")))
    print("\nList of inhabitants:\n")
    master.show()
    print("\nPlaying:\n")
    master.play()

    print("\n\nQuickhand is promoted to wizard.\n")
    master.change_role("Quickhand", Wizard())
    print("\nList of inhabitants:\n")
    master.show()
    print("\nPlaying:\n")
    master.play()


if __name__ == "__main__":
    main()
