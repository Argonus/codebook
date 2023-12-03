class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'Vector({self.x}, {self.y})'
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        return Vector(self.x * other.x, self.y * other.y)

    def scalarMult(self, other):
        return self.x * other.x + self.y * other.y


v1 = Vector(3, 3)
v2 = Vector(1, 2)

v3 = v1 + v2
scalar = v1.scalarMult(v2)

print(v1, v2, v3)
print(scalar)