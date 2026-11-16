import sqlite3

# example of reading the data
conn = sqlite3.connect("data.db")
c = conn.cursor()

c.execute("SELECT * FROM Cars")
rows = c.fetchall()
for row in rows:
    print(row)

conn.close()
