class NumericSystems:
    def __init__(self):
        pass

    # Convert Napier's representation to decimal
    def napierRepresentation(self, number):
        numbers = map(int, [*number])
        powers = map(lambda x: 2 ** x, numbers)
        value = sum(powers)
        return value

    # Then reduce pairs of numbers to a single number
    # a * a = a + 1
    def napierAddition(self, number1, number2):
        num1 = map(int, [*number1])
        num2 = map(int, [*number2])
        num3 = (list(num1) + list(num2))
        num3.sort()

        return num3

    # Then reduce pairs of numbers to a single number
    # a * b = a + b
    def napierMultiplication(self, number1, number2):
        pairs = []

        num1 = list(map(int, [*number1]))
        num2 = list(map(int, [*number2]))

        for i in num1:
            for j in num2:
                pairs.append([i, j])

        return pairs

    # () = 2^ 0
    # (()) = 2 ^ 2 ^ 0 = 2 ^ 1 = 2
    # ((())) = 2 ^ 2 ^ 2 ^ 0 = 2 ^ 2 ^ 1 = 2 ^ 2 = 4
    # ()(()) = 2^0 * 2^1 = 1 + 2 = 3
    # (()((()))) = (2^0 + 2 ^ 2 ^ 2 ^ 0) = (1 + 4) = 2^5 = 32
    def bracketsRepresentation(self, brackets):
        return None

    # 5 | 3 = 5 * 5 * 5 = 125
    # 3 || 3 = 3 | (3 x 3 x 3) = (3 x 3 x 3) x (3 x 3 x 3) x (3 x 3 x 3)
    def arrowsRepresentation(self, arrows):
        return None

    def binaryWithBias(self, number, bias):
        numbers = map(int, [*number])
        numbers = list(numbers).reverse()
        value = decimal - bias

        return numbers
