
class Interpolation:
    def __init__(self, x_points, y_points, debug = False):
        self.x_points = x_points
        self.y_points = y_points
        self.debug = debug

    # Used in advent_of_code_2023/day_09/solution.py
    def lagrange(self, x):
        y = 0
        n = len(self.y_points)

        for i in range(n):
            xi, yi = self.x_points[i], self.y_points[i]
            term = yi
            for j in range(n):
                if i == j:
                    continue

                xj = self.x_points[j]
                term *= (x - xj) / (xi - xj)
            y += term
        return y

    def newton(self, x):
        n = len(self.x_points)
        coef = self.__newton_coefficients(self.x_points[:], self.y_points[:])

        y = coef[0][0]
        temp = 1
        for i in range(1, n):
            temp *= (x - self.x_points[i - 1])
            y += coef[0][i] * temp
            self.print_debug(y)
        return y

    def __newton_coefficients(self, x, y):
        n = len(x)
        # Using None to raise error if the user tries to access a non-initialized value.
        coef = [[None for _ in range(n)] for __ in range(n)]
        # First column is y
        for i in range(n):
            coef[i][0] = y[i]

        self.print_debug(coef)

        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

        self.print_debug(coef)

        return coef

    def print_debug(self, data):
        if self.debug:
            print(data)