import sympy as sp
import logging


class ContinuedFractions:
    def __init__(self, func, debug=False):
        self.func = func
        self.debug = debug

    def findChain(self, num):
        chain = self.__doFindChain([], num)
        return " ".join(map(str, chain))

    def __doFindChain(self, chain, d):
        x = self.func(d)
        r = sp.floor(x)
        chain.append(r)
        # If the number is a perfect square, return the chain.
        if r * r == d:
            return chain
        # Otherwise, go through the process of finding the chain.
        else:
            # Use quadratic surds to verify that period is completed.
            a, p, q = r, 0, 1
            while True:
                self.print_debug(p, q, a)
                p = a * q - p
                q = (d - p * p) // q
                a = (r + p) // q
                chain.append(a)
                if q == 1:
                    self.print_final(p, q, a)
                    break
            return chain

    def print_debug(self, p, q, a):
        if self.debug:
            print("Pk: ", p, "Qk: ", q, "Ak: ", a)

    def print_final(self, p, q, a):
        if self.debug:
            print("Pf: ", p, "Qf: ", q, "Af: ", a)
