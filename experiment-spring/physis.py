import numpy

class defsys:
    def __init__(self, γ=0.1, C=0):
        self.γ = γ
        self.C = C

    def V(self, x, t):
        return 0.5 * x ** 2 - self.C

    def Vq(self, x, t):
        return x

    def f(self, t):
        return self.γ

    def F(self, z, t):
        return self.f(t) * s

    def Fz(self, z, t):
        return self.f(t)