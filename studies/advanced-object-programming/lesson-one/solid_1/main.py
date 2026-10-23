
from gardener import Gardener
from knight import Knight
from master import Master


def main():
    print("\n\nSOLID exercise 1\n")
    master = Master()
    master.set_hero(Gardener("Slenderwillow","f","Oldoak","Sunnydaisy"))
    master.set_hero(Knight("Quickhand","m","Bravedeed","Sunbell","sword","horse"))
    master.set_hero(Knight("Truefriend","m","Highspirit", "Mistymorning","spear","dragon"))
    master.set_hero(Gardener("Greenleaf","m","Brownleaf","Goldendaisy"))
    master.show()
    master.play()

if __name__ == "__main__":
    main()
