r"""
Contact integrators defined for an Hamiltonian of the form
$$
  H(q, p, s) = \frac{p^2}2 + f(t)s^a + V(q,t),
$$
or, equivalently, a Lagrangian of the form
$$
  L(q, \dot{q}, s) = \frac{\dot{q}^2}2 - f(t)s^a - V(q,t).
$$
Here $a\in\{1,2\}$.

These systems are provided via a class providing the functions

- V(q,t) -- the potential (this can include a forcing term),
- Vq(q,t) -- d/dq V(q,t),
- f(t) -- the time-dependent damping.
"""

import numpy as np
import warnings
from scipy.optimize import fsolve


def variational_step_quadratic(system, dt, p, x, s, t):
    r"""
    Contact integrators defined for an Hamiltonian of the form
    $$
      H(q, p, s) = \frac{p^2}2 + f(t)s^2 + V(q,t),
    $$
    or, equivalently, a Lagrangian of the form
    $$
      L(q, \dot{q}, s) = \frac{\dot{q}^2}2 - f(t)s^2 - V(q,t).
    $$

    These systems are provided via a class providing the functions

    - V(q,t) -- the potential (this can include a forcing term),
    - Vq(q,t) -- d/dq V(q,t),
    - f(t) -- the time-dependent damping.
    """
    tnew = t + dt
    xnew = (
        x + (dt - 0.5 * dt ** 2 * system.f(t) * s) * p - 0.5 * dt ** 2 * system.Vq(x, t)
    )
    (snew,) = fsolve(
        lambda snew: s
        - snew
        + 0.5 * dt * ((xnew - x) / dt) ** 2
        - 0.5
        * dt
        * (
            system.V(x, t)
            + system.V(xnew, tnew)
            + 0.5 * (system.f(t) * s ** 2 + system.f(tnew) * snew ** 2)
        ),
        [s],
    )
    pnew = (
        (1.0 - 0.5 * dt * system.f(t) * s) * p
        - 0.5 * dt * (system.Vq(x, t) + system.Vq(xnew, tnew))
    ) / (1.0 + 0.5 * dt * system.f(tnew) * snew)
    return pnew, xnew, snew, tnew


def variational_implicit_step(system, dt, p, x, z, t):
    """
    For Lagrangian functions of the form
    L(j) = 1/(2h^2) (x_{j+1} - x_j)^2 - 1/2 (V(x_j) + V(x_{j+1})) - 1/2 (F(z_j) + F(z_{j+1}))
    """

    tnew = t + dt
    xnew = (
        x + (dt - 0.5 * dt ** 2 * system.Fz(z, t)) * p - 0.5 * dt ** 2 * system.Vq(x, t)
    )
    (znew,) = fsolve(
        lambda znew: z
        - znew
        + 0.5 * dt * np.linalg.norm((xnew - x) / dt) ** 2
        - 0.5
        * dt
        * (
            system.V(x, t)
            + system.V(xnew, tnew)
            + system.F(z, t)
            + system.F(znew, tnew)
        ),
        [z],
    )
    pnew = (
        (1.0 - 0.5 * dt * system.Fz(z, t)) * p
        - 0.5 * dt * (system.Vq(x, t) + system.Vq(xnew, tnew))
    ) / (1.0 + 0.5 * dt * system.Fz(znew, tnew))
    return pnew, xnew, znew, tnew


def variational_step(system, dt, p, x, z, t):
    """
    For Lagrangian functions of the form
    L(j) = 1/(2h^2) (x_{j+1} - x_j)^2 - 1/2 (V(x_j) + V(x_{j+1})) - 1/2 (F(z_j) + F(z_{j+1}))
    """

    tnew = t + dt
    xnew = (
        x + (dt - 0.5 * dt ** 2 * system.Fz(z, t)) * p - 0.5 * dt ** 2 * system.Vq(x, t)
    )
    znew = (
        z
        + 0.5 * dt * np.linalg.norm((xnew - x) / dt) ** 2
        - 0.5 * dt * (system.V(x, t) + system.V(xnew, tnew) + 2 * system.F(z, t))
    )
    pnew = (
        (1.0 - 0.5 * dt * system.Fz(z, t)) * p
        - 0.5 * dt * (system.Vq(x, t) + system.Vq(xnew, tnew))
    ) / (1.0 + 0.5 * dt * system.Fz(znew, tnew))
    return pnew, xnew, znew, tnew


def variational_step_linear(system, dt, p, x, s, t):
    # TODO: add snew = s + dt * L
    tnew = t + dt
    xnew = x + (dt - 0.5 * dt ** 2 * system.f(t)) * p - 0.5 * dt ** 2 * system.Vq(x, t)
    pnew = (1.0 - 0.5 * dt * system.f(t)) / (
        1.0 + 0.5 * dt * system.f(t)
    ) * p + 0.5 * dt * (-system.Vq(x, t) - system.Vq(xnew, tnew)) / (
        1.0 + 0.5 * dt * system.f(t)
    )
    return pnew, xnew, 0, tnew


#
#  Hamiltonian integrator
#


def D(system, dt, p, q, s, t):
    t += dt
    return p, q, s, t


def C(system, dt, p, q, s, t):
    q += p * dt
    s += dt * np.linalg.norm(p) ** 2 / 2.0
    return p, q, s, t


def B(system, dt, p, q, s, t):
    p += -dt * system.Vq(q, t)
    s += -dt * system.V(q, t)
    return p, q, s, t


def A(system, dt, p, q, s, t):
    testdt = 1000
    #if s < 0:
    #    testdt = -2.0 / (s * system.γ)
    #if dt > testdt:
    #    warnings.warn(
    #        f"The integration reached a singularity. Time step: ${dt}$ , singulrity in ${testdt}$"
    #    )
    #    s = float("-inf")
    #    p = float("+inf")
    #else:
    s *= 2.0 / (2.0 + s * dt * system.γ)
    p *= 4.0 / (2.0 + s * dt * system.γ) ** 2
    # To test compositional
    # p += -dt * system.γ * p * s
    # s += -dt * system.γ * s**2 / 2.0
    return p, q, s, t


cbabc = [(D, 0.5), (C, 0.5), (B, 0.5), (A, 1), (B, 0.5), (C, 0.5), (D, 0.5)]
bcacb = [(D, 0.5), (B, 0.5), (C, 0.5), (A, 1), (C, 0.5), (B, 0.5), (D, 0.5)]
abcba = [(D, 0.5), (A, 0.5), (B, 0.5), (C, 1), (B, 0.5), (A, 0.5), (D, 0.5)]
bacab = [(D, 0.5), (B, 0.5), (A, 0.5), (C, 1), (A, 0.5), (B, 0.5), (D, 0.5)]
cabac = [(D, 0.5), (C, 0.5), (A, 0.5), (B, 1), (A, 0.5), (C, 0.5), (D, 0.5)]
acbca = [(D, 0.5), (A, 0.5), (C, 0.5), (B, 1), (C, 0.5), (A, 0.5), (D, 0.5)]

mappers = [cbabc, bcacb, abcba, bacab, cabac, acbca]


def step1q(system, dt, p, q, s, t, mapper=cbabc):
    """
    Step for contact splitting integrator, quadratic in s
    """
    for ap, coeff in mapper:
        p, q, s, t = ap(system, dt * coeff, p, q, s, t)
    return p, q, s, t


def step1l(system, dt, p, q, s, t):
    """
    Step for contact splitting integrator, linear in s.
    Hardcoded version DCBABCD
    """
    # dt/2 D
    t += dt / 2.0

    # dt/2 C
    q += p * dt / 2.0
    s += np.linalg.norm(p) ** 2 * dt / 4.0

    # dt/2 B
    p -= system.Vq(q, t) * dt / 2
    s -= system.V(q, t) * dt / 2

    # dt A
    etf = np.exp(-dt * system.f(t))
    p *= etf
    s *= etf

    # dt/2 B
    p -= system.Vq(q, t) * dt / 2
    s -= system.V(q, t) * dt / 2

    # dt/2 C
    q += p * dt / 2
    s += np.linalg.norm(p) ** 2 * dt / 4

    # dt/2 D
    t += dt / 2

    return (p, q, s, t)


_A = [-1.17767998417887, 0.235573213359357, 0.784513610477560]
_A0 = 1 - 2 * sum(_A)
a_six = _A[::-1] + [_A0] + _A[:]

_B = [-2.13228522200144, 0.00426068187079180, 1.43984816797678]
_B0 = 1 - 2 * sum(_B)
b_six = _B[::-1] + [_B0] + _B[:]

_C = [0.00152886228424922, -2.14403531630539, 1.44778256239930]
_C0 = 1 - 2 * sum(_C)
c_six = _C[::-1] + [_C0] + _C[:]

x0, x1 = np.array([-np.cbrt(2), 1]) / (2 - np.cbrt(2))
z0, z1 = np.array([-np.power(2, 1 / 5), 1]) / (2 - np.power(2, 1 / 5))
_E = [x0 * z0, x1 * z0, x1 * z1, x0 * z1, x1 * z1]
e_six = _E[-1:0:-1] + _E[:]


def step6(system, dt, p, q, s, t, a=a_six, stepper=step1l):
    for coeff in a:
        p, q, s, t = stepper(system, coeff * dt, p, q, s, t)
    return p, q, s, t


def step6e(system, dt, p, q, s, t):
    return step6(system, dt, p, q, s, t, a=e_six)


def step6b(system, dt, p, q, s, t):
    return step6(system, dt, p, q, s, t, a=b_six)


def step6c(system, dt, p, q, s, t):
    return step6(system, dt, p, q, s, t, a=c_six)


def step6q(system, dt, p, q, s, t):
    return step6(system, dt, p, q, s, t, stepper=step1q)


def discrete_lag4(system, x0, x1, z0, t, dt):
    """returns the discrete Lagrangian and its derivative as a tuple (lag, d(lag)/d(x0), d(lag)/d(x1), d(lag)/d(z0))"""

    def z1(b):
        k1 = dt * system.lag(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        k2 = dt * system.lag(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
        k3 = dt * system.lag(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
        k4 = dt * system.lag(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
        return z0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

    def z1b(b):
        k1 = dt * system.lag(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        k2 = dt * system.lag(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
        k3 = dt * system.lag(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
        # k4 = dt*system.lag(x1, (x0+3*x1-4*b)/dt, z0 + k3, t + dt)
        kb1 = 4 * system.lagv(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        kb2 = (
            dt * system.lagq(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2) * kb1
        )
        kb3 = (
            dt * system.lagq(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2) * kb2
        )
        kb4 = (
            -4 * system.lagv(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
            + dt * system.lagz(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt) * kb3
        )
        return kb1 / 6 + kb2 / 3 + kb3 / 3 + kb4 / 6

    def z1x0(b):
        k1 = dt * system.lag(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        k2 = dt * system.lag(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
        k3 = dt * system.lag(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
        # k4 = dt*system.lag(x1, (x0+3*x1-4*b)/dt, z0 + k3, t + dt)
        kx01 = dt * system.lagq(
            x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t
        ) - 3 * system.lagv(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        kx02 = (
            -system.lagv(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2) * kx01
        )
        kx03 = (
            -system.lagv(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2) * kx02
        )
        kx04 = (
            system.lagv(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
            + dt * system.lagz(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt) * kx03
        )
        return kx01 / 6 + kx02 / 3 + kx03 / 3 + kx04 / 6

    def z1x1(b):
        k1 = dt * system.lag(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        k2 = dt * system.lag(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
        k3 = dt * system.lag(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
        # k4 = dt*system.lag(x1, (x0+3*x1-4*b)/dt, z0 + k3, t + dt)
        kx11 = -system.lagv(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        kx12 = (
            system.lagv(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2) * kx11
        )
        kx13 = (
            system.lagv(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
            + 0.5 * dt * system.lagz(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2) * kx12
        )
        kx14 = (
            dt * system.lagq(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
            + 3 * system.lagv(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
            + dt * system.lagz(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt) * kx13
        )
        return kx11 / 6 + kx12 / 3 + kx13 / 3 + kx14 / 6

    def z1z0(b):
        k1 = dt * system.lag(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        k2 = dt * system.lag(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
        k3 = dt * system.lag(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
        # k4 = dt*system.lag(x1, (x0+3*x1-4*b)/dt, z0 + k3, t + dt)
        kz01 = dt * system.lagz(x0, (-3 * x0 - x1 + 4 * b) / dt, z0, t)
        kz02 = (
            dt
            * system.lagz(b, (x1 - x0) / dt, z0 + k1 / 2, t + dt / 2)
            * (1 + kz01 / 2)
        )
        kz03 = (
            dt
            * system.lagz(b, (x1 - x0) / dt, z0 + k2 / 2, t + dt / 2)
            * (1 + kz02 / 2)
        )
        kz04 = (
            dt
            * system.lagz(x1, (x0 + 3 * x1 - 4 * b) / dt, z0 + k3, t + dt)
            * (1 + kz03)
        )
        return 1.0 + kz01 / 6 + kz02 / 3 + kz03 / 3 + kz04 / 6

    bcrit = fsolve(z1b, (x0 + x1) / 2)
    return (z1(bcrit) - z0, z1x0(bcrit), z1x1(bcrit), z1z0(bcrit) - 1.0)


def step_lag4(system, dt, p, q, s, t):
    def lag(x0, x1, z0):
        return discrete_lag4(system, x0, x1, z0, t, dt)[0]

    def lagx0(x0, x1, z0):
        return discrete_lag4(system, x0, x1, z0, t, dt)[1]

    def lagx1(x0, x1, z0):
        return discrete_lag4(system, x0, x1, z0, t, dt)[2]

    def lagz0(x0, x1, z0):
        return discrete_lag4(system, x0, x1, z0, t, dt)[3]

    qnew = fsolve(lambda x1: lagx0(q, x1, s) + p * (1.0 + lagz0(q, x1, s)), q + dt * p)
    pnew = lagx1(q, qnew, s)
    snew = s + lag(q, qnew, s)
    return (pnew, qnew, snew, t + dt)


def integrate(stepper, system, tspan, p0, q0, s0, ttol=1e-10):
    """
    Integrate [system] with initial conditions [p0], [q0]
    using the hamiltonian integrator provided in step.

    [tspan] is usually [np.linspace(t0, tfinal, num=steps)]
    """
    dt = tspan[1] - tspan[0]
    steps = len(tspan)
    init = [p0, q0]

    solpq = np.empty([steps, *np.shape(init)], dtype=np.float64)
    sols = np.empty(steps, dtype=np.float64)
    solpq[0] = np.array(init)
    sols[0] = s0

    for i in range(steps - 1):
        p, q = np.copy(solpq[i])
        s = sols[i]
        t = tspan[i]
        pnew, qnew, snew, tnew = stepper(system, dt, p, q, s, t)
        if abs(tnew - t - dt) > ttol:
            warnings.warn(f"tnew-t-dt, dt inconsistency: {tnew-t-dt}, {dt}")
        solpq[i + 1] = [pnew, qnew]
        sols[i + 1] = snew

    return solpq, sols, tspan
