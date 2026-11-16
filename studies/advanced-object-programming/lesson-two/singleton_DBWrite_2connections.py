import sqlite3


class DatabaseConnection:
    """Singleton zarządzający wspólnym połączeniem z bazą SQLite.

    Aplikacja jest jednowątkowa, więc proste _instance wystarcza
    (bez dodatkowej blokady dla wątków).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connection = sqlite3.connect("data.db")
        return cls._instance

    def get_connection(self):
        return self._connection

    def commit(self):
        self._connection.commit()

    def close(self):
        self._connection.close()


# Część pierwsza: tworzy tabelę przez wspólne połączenie
first = DatabaseConnection()
conn1 = first.get_connection()
c = conn1.cursor()
print("Database Objects DB1", conn1)

c.execute('''CREATE TABLE Cars(Id int, Name text, Type text, Price int)''')


# Część druga: tej samej instancji używa "inna" część programu
second = DatabaseConnection()
conn2 = second.get_connection()
c2 = conn2.cursor()
print("Database Objects DB2", conn2)

c2.execute("INSERT INTO Cars VALUES(1,'Audi A3','sedan',52642)")

cars = [('22', 'Ford Focus', 'combi', 4321),
        ('22', 'Opel Astra', 'hatchback', 5500),
        ('22', 'Toyota Auris', 'hatchback', 6574)]
c2.executemany('INSERT INTO Cars VALUES (?,?,?,?)', cars)

# Sprawdzenie z pkt. 3: te same obiekty Singletona i to samo połączenie
print("Singletons identical:", first is second)
print("Connections identical:", conn1 is conn2)

# Zatwierdzenie i zamknięcie raz, po wykonaniu wszystkich operacji (pkt. 4)
second.commit()
second.close()
