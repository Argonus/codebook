from abc import ABC, abstractmethod


class Role(ABC):
    name = ""

    @abstractmethod
    def describe(self): ...

    @abstractmethod
    def play(self): ...
