from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Function2D(ABC):
    name = ""

    @abstractmethod
    def evaluate(self, x, y): ...


class Sphere(Function2D):
    name = "Sphere"

    def evaluate(self, x, y):
        return x * x + y * y


class Matyas(Function2D):
    name = "Matyas"

    def evaluate(self, x, y):
        return 0.26 * (x * x + y * y) - 0.48 * x * y


class Rastrigin(Function2D):
    name = "Rastrigin"

    def evaluate(self, x, y):
        return 20 + x * x - 10 * np.cos(2 * np.pi * x) + y * y - 10 * np.cos(2 * np.pi * y)


class FunctionDecorator(Function2D):
    def __init__(self, function):
        self.function = function
        self.name = function.name

    def evaluate(self, x, y):
        return self.function.evaluate(x, y)


class ValueDecorator(FunctionDecorator):
    def evaluate(self, x, y):
        value = super().evaluate(x, y)
        if np.ndim(value) == 0:
            print(f"{self.name}({x:g}, {y:g}) = {value:g}")
        return value


class PlotDecorator(FunctionDecorator):
    def __init__(self, function, x_range, y_range, samples=200):
        super().__init__(function)
        self.x_range = x_range
        self.y_range = y_range
        self.samples = samples
        self.figure = None

    def evaluate(self, x, y):
        value = super().evaluate(x, y)
        if np.ndim(value) != 0:
            return value

        X, Y = np.meshgrid(
            np.linspace(*self.x_range, self.samples),
            np.linspace(*self.y_range, self.samples),
        )
        Z = super().evaluate(X, Y)
        self.figure = plt.figure()
        axes = plt.axes(projection="3d")
        axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", linewidth=0, antialiased=False)
        return value


def main():
    x = 0.5
    y = 0.5

    sphere = Sphere()
    sphere = PlotDecorator(sphere, x_range=(-20, 20), y_range=(-20, 20))
    sphere = ValueDecorator(sphere)
    sphere.evaluate(x, y)
    plt.savefig("sphere.png")

    matyas = Matyas()
    matyas = PlotDecorator(matyas, x_range=(-20, 20), y_range=(-20, 20))
    matyas = ValueDecorator(matyas)
    matyas.evaluate(x, y)
    plt.savefig("matyas.png")

    rastrigin = Rastrigin()
    rastrigin = PlotDecorator(rastrigin, x_range=(-20, 20), y_range=(-20, 20))
    rastrigin = ValueDecorator(rastrigin)
    rastrigin.evaluate(x, y)
    plt.savefig("rastrigin.png")

    plt.show()


if __name__ == "__main__":
    main()
